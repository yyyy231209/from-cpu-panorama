#include "mjpeg_streamer.h"
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

MjpegStreamer::MjpegStreamer(int port) : port_(port) {}

MjpegStreamer::~MjpegStreamer() { stop(); }

void MjpegStreamer::start() {
    running_.store(true);
    server_thread_ = std::thread(&MjpegStreamer::server_loop, this);
}

void MjpegStreamer::stop() {
    running_.store(false);
    jpeg_cv_.notify_all();
    if (server_fd_ >= 0) shutdown(server_fd_, SHUT_RDWR);
    if (server_thread_.joinable()) server_thread_.join();
}

void MjpegStreamer::push_frame(const cv::Mat& frame) {
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};
    {
        std::lock_guard<std::mutex> lk(jpeg_mtx_);
        cv::imencode(".jpg", frame, jpeg_buf_, params);
        new_frame_ = true;
    }
    jpeg_cv_.notify_one();
}

void MjpegStreamer::server_loop() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        printf("[stream] socket fail: %s\n", strerror(errno));
        return;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        printf("[stream] bind fail: %s\n", strerror(errno));
        close(server_fd_);
        return;
    }

    if (listen(server_fd_, 2) < 0) {
        printf("[stream] listen fail: %s\n", strerror(errno));
        close(server_fd_);
        return;
    }

    printf("[stream] MJPEG server on port %d\n", port_);

    while (running_.load()) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            break;
        }

        printf("[stream] client: %s\n", inet_ntoa(client_addr.sin_addr));

        const char* header =
            "HTTP/1.0 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "\r\n";
        send(client_fd, header, strlen(header), MSG_NOSIGNAL);

        bool ok = true;
        while (ok && running_.load()) {
            std::vector<uchar> jpeg_copy;
            {
                std::unique_lock<std::mutex> lk(jpeg_mtx_);
                jpeg_cv_.wait_for(lk, std::chrono::milliseconds(500),
                                  [this]{ return new_frame_ || !running_.load(); });
                if (!running_.load()) break;
                if (!new_frame_) continue;
                jpeg_copy = jpeg_buf_;
                new_frame_ = false;
            }

            char part_header[256];
            int hdr_len = snprintf(part_header, sizeof(part_header),
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                jpeg_copy.size());
            ssize_t s1 = send(client_fd, part_header, hdr_len, MSG_NOSIGNAL);
            ssize_t s2 = send(client_fd, jpeg_copy.data(), jpeg_copy.size(), MSG_NOSIGNAL);
            if (s1 < 0 || s2 < 0) ok = false;
        }

        close(client_fd);
    }

    close(server_fd_);
}
