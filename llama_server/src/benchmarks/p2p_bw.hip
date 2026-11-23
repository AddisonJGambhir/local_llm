#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  printf("ERR %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e)); return 1;} } while(0)

static double gbps(size_t bytes, double sec, int iters){
  return (double)bytes * iters / sec / 1e9;
}

int main(){
  int n=0; CK(hipGetDeviceCount(&n));
  printf("visible devices: %d\n", n);
  for(int i=0;i<n;i++){
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p,i));
    printf("  dev%d: %-28s arch=%s  %.1f GiB\n", i, p.name, p.gcnArchName,
           p.totalGlobalMem/1073741824.0);
  }
  if(n<2){ printf("need 2 devices\n"); return 1; }

  const size_t SZ = 256ull<<20;   // 256 MiB
  const int ITERS = 20;

  for(int a=0;a<n;a++) for(int b=0;b<n;b++){
    if(a==b) continue;
    int can=0; CK(hipDeviceCanAccessPeer(&can,a,b));
    printf("canAccessPeer %d->%d : %s\n", a,b, can?"YES":"NO");
  }

  void *pa,*pb; void* hbuf;
  CK(hipSetDevice(0)); CK(hipMalloc(&pa,SZ));
  CK(hipSetDevice(1)); CK(hipMalloc(&pb,SZ));
  CK(hipHostMalloc(&hbuf,SZ,hipHostMallocDefault));

  // try to turn on peer access both ways
  int peer_ok = 1;
  CK(hipSetDevice(0));
  if(hipDeviceEnablePeerAccess(1,0)!=hipSuccess) peer_ok=0;
  CK(hipSetDevice(1));
  if(hipDeviceEnablePeerAccess(0,0)!=hipSuccess) peer_ok=0;
  printf("peer access enabled: %s\n", peer_ok?"YES":"NO (falls back to host staging)");

  auto bench=[&](const char* tag, auto fn){
    fn(); CK(hipDeviceSynchronize());                 // warmup
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<ITERS;i++) fn();
    CK(hipDeviceSynchronize());
    auto t1=std::chrono::steady_clock::now();
    double s=std::chrono::duration<double>(t1-t0).count();
    printf("  %-34s %7.2f GB/s\n", tag, gbps(SZ,s,ITERS));
    return 0;
  };

  printf("\n-- device-to-device (the TP all-reduce path) --\n");
  bench("dev0 -> dev1  (XTX -> R9700)", [&]{ hipMemcpyPeer(pb,1,pa,0,SZ); });
  bench("dev1 -> dev0  (R9700 -> XTX)", [&]{ hipMemcpyPeer(pa,0,pb,1,SZ); });

  printf("\n-- host round-trip, for reference --\n");
  CK(hipSetDevice(0));
  bench("dev0 -> host (D2H)", [&]{ hipMemcpy(hbuf,pa,SZ,hipMemcpyDeviceToHost); });
  bench("host -> dev0 (H2D)", [&]{ hipMemcpy(pa,hbuf,SZ,hipMemcpyHostToDevice); });
  CK(hipSetDevice(1));
  bench("dev1 -> host (D2H)  [chipset x4]", [&]{ hipMemcpy(hbuf,pb,SZ,hipMemcpyDeviceToHost); });
  bench("host -> dev1 (H2D)  [chipset x4]", [&]{ hipMemcpy(pb,hbuf,SZ,hipMemcpyHostToDevice); });
  return 0;
}
