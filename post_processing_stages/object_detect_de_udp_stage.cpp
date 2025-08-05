/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2021, Raspberry Pi (Trading) Limited
 *
 * object_detect_udp.cpp - UDP detection results
 */

#include "opencv2/imgproc.hpp"

#include "core/rpicam_app.hpp"

#include "post_processing_stages/post_processing_stage.hpp"

#include "object_detect.hpp"

#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace cv;


using Rectange = libcamera::Rectangle;
using Stream = libcamera::Stream;

// Define the start delimiter for our binary protocol
constexpr static uint32_t START_DELIMITER = 0xDDCCBBAA; // little-endian representation of 0xAA, 0xBB, 0xCC, 0xDD
#define UDP_IP_DEFAULT false
#define UDP_IP "127.0.0.1" 
#define UDP_PORT 12347

class ObjectDetect_DE_UDPStage : public PostProcessingStage
{
public:
	ObjectDetect_DE_UDPStage(RPiCamApp *app) : PostProcessingStage(app), sockfd_(-1) {}

	char const *Name() const override;

	void Read(boost::property_tree::ptree const &params) override;

	void Configure() override;
	
	bool Process(CompletedRequestPtr &completed_request) override;

	virtual ~ObjectDetect_DE_UDPStage() override;
	
private:
	Stream *stream_;
	std::string udp_broadcast_address = "127.0.0.1";
	u_int16_t udp_broadcast_port = 12345;
	int sockfd_;
    struct sockaddr_in servaddr_;
};

#define NAME "object_detect_de_udp"

char const *ObjectDetect_DE_UDPStage::Name() const
{
	return NAME;
}


ObjectDetect_DE_UDPStage::~ObjectDetect_DE_UDPStage()
{
    if (sockfd_ != -1)
    {
        close(sockfd_);
        std::cerr << "UDP socket closed." << std::endl;
    }
}


void ObjectDetect_DE_UDPStage::Configure()
{
	stream_ = app_->GetMainStream();
	
	// Initialize UDP socket
    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0)
    {
        perror("UDP socket creation failed");
        // Handle error, perhaps throw an exception or exit
        return;
    }

    memset(&servaddr_, 0, sizeof(servaddr_));

    // Filling server information
    servaddr_.sin_family = AF_INET;
    servaddr_.sin_port = htons(udp_broadcast_port);
    if (inet_pton(AF_INET, udp_broadcast_address.c_str(), &servaddr_.sin_addr) <= 0)
    {
        perror("Invalid address/ Address not supported");
        close(sockfd_);
        sockfd_ = -1; // Mark as invalid
        return;
    }

    std::cerr << "UDP socket initialized for IP: " << udp_broadcast_address << ", Port: " << udp_broadcast_port << std::endl;
}


void ObjectDetect_DE_UDPStage::Read(boost::property_tree::ptree const &params)
{
	udp_broadcast_address = params.get<std::string>("ip", UDP_IP);
	udp_broadcast_port = params.get<u_int16_t>("port", UDP_PORT);
}

bool ObjectDetect_DE_UDPStage::Process(CompletedRequestPtr &completed_request)
{
	if (!stream_)
        return false;

    BufferWriteSync w(app_, completed_request->buffers[stream_]);
	libcamera::Span<uint8_t> buffer = w.Get()[0];
	uint32_t *ptr = (uint32_t *)buffer.data();
    StreamInfo info = app_->GetStreamInfo(stream_);

    std::vector<Detection> detections;
    completed_request->post_process_metadata.Get("object_detect.results", detections);

    Mat image(info.height, info.width, CV_8U, ptr, info.stride);
	const unsigned int video_width = info.width;
	const unsigned int video_height = info.height;

	
    if (sockfd_ == -1)
        return false;
    
    for (auto &detection : detections)
	{
		// Draw rectangle and text on the image
		std::stringstream text_stream;
		text_stream << detection.name << " " << (int)(detection.confidence * 100) << "%";
		std::string text = text_stream.str();
		
        // Use a vector to dynamically build the binary message
        std::vector<char> udp_data_buffer;

        // 1. Add start delimiter (4 bytes)
        udp_data_buffer.insert(udp_data_buffer.end(), (char*)&START_DELIMITER, (char*)&START_DELIMITER + sizeof(START_DELIMITER));

        // 2. Add x, y, width, height (8 bytes each)
        const double x = (double)detection.box.x / (double)video_width;
		const double y = (double)detection.box.y / (double)video_height;
		const double width = (double)detection.box.width / (double)video_width;
		const double height = (double)detection.box.height / (double)video_height;

		udp_data_buffer.insert(udp_data_buffer.end(), (char*)&x, (char*)&x + sizeof(x));
		udp_data_buffer.insert(udp_data_buffer.end(), (char*)&y, (char*)&y + sizeof(y));
		udp_data_buffer.insert(udp_data_buffer.end(), (char*)&width, (char*)&width + sizeof(width));
		udp_data_buffer.insert(udp_data_buffer.end(), (char*)&height, (char*)&height + sizeof(height));

        // 3. Add name length and name string
        uint32_t name_length = detection.name.length();
        if (name_length > 255) {
            // Truncate or handle error for names longer than 255 chars
            name_length = 255;
        }
        udp_data_buffer.push_back(name_length);
        udp_data_buffer.insert(udp_data_buffer.end(), detection.name.begin(), detection.name.begin() + name_length);

        // 4. Add confidence (4 bytes)
        const float confidence = detection.confidence;
        udp_data_buffer.insert(udp_data_buffer.end(), (char*)&confidence, (char*)&confidence + sizeof(confidence));

        // Send data via UDP
        const ssize_t bytes_sent = sendto(sockfd_, udp_data_buffer.data(), udp_data_buffer.size(), 0,
                                        (const struct sockaddr *)&servaddr_, sizeof(servaddr_));
        if (bytes_sent < 0)
        {
            perror("Failed to send UDP message");
        }
        
	}

	return false;
}

static PostProcessingStage *Create(RPiCamApp *app)
{
	return new ObjectDetect_DE_UDPStage(app);
}

static RegisterStage reg(NAME, &Create);
