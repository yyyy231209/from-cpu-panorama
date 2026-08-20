#include "worker_cpu.h"
#include "cpu_queue.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <linux/videodev2.h>

struct V4L2Buf {
    void* start = nullptr;
    size_t length = 0;
};

static int xioctl(int fd, int req, void* arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

static int cam_init(const char* dev, int& fd, V4L2Buf* bufs, int& nbufs) {
    fd = open(dev, O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        printf("open %s fail: %s\n", dev, strerror(errno));
        return -1;
    }

    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 320;
    fmt.fmt.pix.height = 240;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        printf("VIDIOC_S_FMT fail: %s\n", strerror(errno));
        close(fd); fd = -1; return -1;
    }

    struct v4l2_requestbuffers req = {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        printf("VIDIOC_REQBUFS fail: %s\n", strerror(errno));
        close(fd); fd = -1; return -1;
    }
    nbufs = req.count;

    for (int i = 0; i < nbufs; i++) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            printf("VIDIOC_QUERYBUF fail: %s\n", strerror(errno));
            close(fd); fd = -1; return -1;
        }
        bufs[i].length = buf.length;
        bufs[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (bufs[i].start == MAP_FAILED) {
            printf("mmap fail: %s\n", strerror(errno));
            close(fd); fd = -1; return -1;
        }
        if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            printf("VIDIOC_QBUF fail: %s\n", strerror(errno));
            close(fd); fd = -1; return -1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        printf("VIDIOC_STREAMON fail: %s\n", strerror(errno));
        close(fd); fd = -1; return -1;
    }

    printf("[cam] %s opened, %d bufs\n", dev, nbufs);
    return 0;
}

static void cam_deinit(int& fd, V4L2Buf* bufs, int nbufs) {
    if (fd < 0) return;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < nbufs; i++) {
        if (bufs[i].start && bufs[i].start != MAP_FAILED)
            munmap(bufs[i].start, bufs[i].length);
    }
    close(fd);
    fd = -1;
}

void worker_cpu_func(CPUWorkerCtx& ctx, CPUQueue& queue) {
    int fd = -1;
    V4L2Buf bufs[4];
    int nbufs = 0;

    if (cam_init(ctx.dev_path.c_str(), fd, bufs, nbufs) < 0) {
        printf("[cam%d] init failed\n", ctx.id);
        return;
    }
    ctx.cam_fd = fd;

    int warmup = 1;
    while (warmup-- > 0 && ctx.running.load()) {
        struct pollfd pfd = {fd, POLLIN, 0};
        if (poll(&pfd, 1, 2000) <= 0) continue;
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd, VIDIOC_DQBUF, &buf) < 0) continue;
        xioctl(fd, VIDIOC_QBUF, &buf);
    }

    printf("[cam%d] capturing ...\n", ctx.id);

    while (ctx.running.load()) {
        struct pollfd pfd = {fd, POLLIN, 0};
        int r = poll(&pfd, 1, 1000);
        if (r < 0) break;
        if (r == 0) continue;

        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd, VIDIOC_DQBUF, &buf) < 0) continue;

        cv::Mat raw(1, buf.bytesused, CV_8UC1, bufs[buf.index].start);
        cv::Mat frame = cv::imdecode(raw, cv::IMREAD_COLOR);
        xioctl(fd, VIDIOC_QBUF, &buf);

        if (frame.empty()) continue;
        if (frame.cols != 320 || frame.rows != 240)
            cv::resize(frame, frame, cv::Size(320, 240));

        queue.push(ctx.id, frame);
    }

    cam_deinit(fd, bufs, nbufs);
    printf("[cam%d] stopped\n", ctx.id);
}
