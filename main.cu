extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

__global__ void laplacianKernel(const unsigned char* input, float* output, int width, int height);
__global__ void reductionKernel(float* input, float* output, int size);

class LaplacianVarianceCalculator {
private:
    unsigned char* d_input;
    float* d_output;
    float* d_tempOutput;
    int width, height;
    size_t imageSize, imageBytes, outputBytes;

public:
    LaplacianVarianceCalculator(int w, int h) : width(w), height(h) {
        imageSize = width * height;
        imageBytes = imageSize * sizeof(unsigned char);
        outputBytes = imageSize * sizeof(float);

        cudaMalloc((void**)&d_input, imageBytes);
        cudaMalloc((void**)&d_output, outputBytes);
        cudaMalloc((void**)&d_tempOutput, outputBytes);
    }

    ~LaplacianVarianceCalculator() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_tempOutput);
    }

    float calculateVariance(const unsigned char* h_input) {
        cudaMemcpy(d_input, h_input, imageBytes, cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        laplacianKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaDeviceSynchronize();

        int blocksForReduction = (imageSize + blockSize.x - 1) / blockSize.x;
        reductionKernel<<<blocksForReduction, blockSize, blockSize.x * sizeof(float)>>>(d_output, d_tempOutput, imageSize);
        cudaDeviceSynchronize();

        float* h_tempOutput = new float[blocksForReduction];
        cudaMemcpy(h_tempOutput, d_tempOutput, blocksForReduction * sizeof(float), cudaMemcpyDeviceToHost);

        float totalLaplacian = 0;
        for (int i = 0; i < blocksForReduction; i++) {
            totalLaplacian += h_tempOutput[i];
        }
        delete[] h_tempOutput;
        // Add padding and variance
        return totalLaplacian;
    }
};

__global__ void laplacianKernel(const unsigned char* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        float laplacian = input[(y - 1) * width + x] +
                          input[(y + 1) * width + x] +
                          input[y * width + (x - 1)] +
                          input[y * width + (x + 1)] -
                          4 * input[y * width + x];
        output[y * width + x] =  laplacian * laplacian;
    }
}

__global__ void reductionKernel(float* input, float* output, int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < size) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

class FrameSaver {
public:
    static void saveFrame(const AVFrame *frame, int frameNumber) {
        std::string fileName = "frame_" + std::to_string(frameNumber) + ".raw";
        std::ofstream outFile(fileName, std::ios::out | std::ios::binary);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open the file for writing: " << fileName << std::endl;
            return;
        }
        for (int y = 0; y < frame->height; y++) {// AV_PIX_FMT_GRAY8 format
            outFile.write(reinterpret_cast<char*>(frame->data[0] + y * frame->linesize[0]), frame->width);
        }
        std::cout << "Saved " << fileName << std::endl;
        outFile.close();
    }
};

unsigned char* get_y_channel(AVFrame* frame, int& width, int& height) {
    width = frame->width;
    height = frame->height;
    return frame->data[0];  //Y channel data
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>\n";
        return -1;
    }

    auto startMain = std::chrono::steady_clock::now();

    av_register_all();
    avcodec_register_all();

    AVFormatContext* pFormatContext = avformat_alloc_context();

    auto start = std::chrono::high_resolution_clock::now();
    if (avformat_open_input(&pFormatContext, argv[1], nullptr, nullptr) != 0) {
        std::cerr << "Could not open video file\n";
        return -1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> open_duration = stop - start;
    std::cout << "Time taken to open file: " << open_duration.count() << " ms\n";


    if (avformat_find_stream_info(pFormatContext, nullptr) < 0) {
        std::cerr << "Could not find stream information\n";
        return -1;
    }

    // Get first video stream
    int videoStreamIndex = -1;
    AVCodecParameters* codecParameters = nullptr;
    for (unsigned i = 0; i < pFormatContext->nb_streams; i++) {
        if (pFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            codecParameters = pFormatContext->streams[i]->codecpar;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "Could not find a video stream\n";
        return -1;
    }

    AVCodec* codec = avcodec_find_decoder(codecParameters->codec_id);
    if (!codec) {
        std::cerr << "Unsupported codec\n";
        return -1;
    }

    AVCodecContext* codecContext = avcodec_alloc_context3(codec);
    if (avcodec_parameters_to_context(codecContext, codecParameters) < 0) {
        std::cerr << "Could not copy codec context\n";
        return -1;
    }

    if (avcodec_open2(codecContext, codec, nullptr) < 0) {
        std::cerr << "Could not open codec\n";
        return -1;
    }

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    int frameCount = 0;

    LaplacianVarianceCalculator lvc(codecContext->width, codecContext->height);

    std::vector<unsigned char*> y_channels;

    auto startDecoding = std::chrono::steady_clock::now();

    // Add lookup for frame writing 
    while (av_read_frame(pFormatContext, packet) >= 0) {
    if (packet->stream_index == videoStreamIndex) {
        if (avcodec_send_packet(codecContext, packet) == 0) {
            while (avcodec_receive_frame(codecContext, frame) == 0) {
                int width, height;
                unsigned char* y_channel = get_y_channel(frame, width, height);
                // frames.push_back(frame);
                y_channels.push_back(y_channel);
            }
        }
    }
    av_packet_unref(packet);
    }

    auto endDecoding = std::chrono::steady_clock::now();
    auto decodingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDecoding - startDecoding);
    std::cout << "Decoding time: " << decodingTime.count() << " ms" << std::endl;

    auto startLvc = std::chrono::steady_clock::now();

    for (auto& y_channel : y_channels) {
        float variance = lvc.calculateVariance(y_channel);
        std::cout << "Variance: " << variance << std::endl;
    }

    auto endLvc = std::chrono::steady_clock::now();
    auto lvcTime = std::chrono::duration_cast<std::chrono::milliseconds>(endLvc - startLvc);
    std::cout << "LVC processing time: " << lvcTime.count() << " ms" << std::endl;

    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codecContext);
    avformat_close_input(&pFormatContext);

    auto endMain = std::chrono::steady_clock::now();
    auto mainTime = std::chrono::duration_cast<std::chrono::milliseconds>(endMain - startMain);
    std::cout << "Entire processing time: " << mainTime.count() << " ms" << std::endl;

    return 0;
}
