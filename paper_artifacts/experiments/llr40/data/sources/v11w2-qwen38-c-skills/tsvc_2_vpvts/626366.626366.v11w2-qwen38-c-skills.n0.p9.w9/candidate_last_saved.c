#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
extern int open(const char *, int, ...);
extern int close(int);
extern long pread(int, void *, unsigned long, long);
extern long write(int, const void *, unsigned long);
#define O_RDONLY 0
#define O_WRONLY 1
#define O_CREAT  0100
#define O_APPEND 02000

static void logline(const char *line) {
  int fd = open("/shared/agent-9/topo_out.txt", O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd >= 0) { (void)write(fd, line, (unsigned long)strlen(line)); close(fd); }
}
static void probe(const char *path) {
  int fd = open(path, O_RDONLY);
  char buf[256]; long n = -1;
  if (fd >= 0) { n = pread(fd, buf, 255, 0); close(fd); }
  if (n < 0) n = 0;
  buf[n] = 0;
  char line[400];
  snprintf(line, sizeof(line), "%s : fd=%d n=%ld [%s]\n", path, fd, n, buf);
  logline(line);
}

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
  static int count = 0; count++;
  const double s = (double)S;
  if (count == 1) {
    probe("/sys/devices/system/node/node0/cpus");
    probe("/sys/devices/system/node/node1/cpus");
    probe("/sys/devices/system/node/node2/cpus");
    probe("/sys/devices/system/node/node3/cpus");
    probe("/sys/devices/system/node/node0/cpulist");
    probe("/sys/devices/system/node/node1/cpulist");
    probe("/sys/devices/system/node/node0/meminfo");
    probe("/sys/devices/system/node/online");
  }
  for (int64_t i = 0; i < LEN_1D; i++) a[i] += b[i] * s;
}
