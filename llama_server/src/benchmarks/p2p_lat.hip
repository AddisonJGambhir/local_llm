#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s: %s\n",#x,hipGetErrorString(e));return 1;}}while(0)
using clk=std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b){
  return std::chrono::duration<double>(b-a).count()*1e6; }

int main(){
  int n; CK(hipGetDeviceCount(&n)); if(n<2){printf("need 2\n");return 1;}
  const size_t MAX = 32ull<<20;
  void *pa,*pb;
  CK(hipSetDevice(0)); CK(hipMalloc(&pa,MAX)); hipDeviceEnablePeerAccess(1,0);
  CK(hipSetDevice(1)); CK(hipMalloc(&pb,MAX)); hipDeviceEnablePeerAccess(0,0);

  printf("payload      per-copy      eff BW     what it is\n");
  printf("-----------------------------------------------------------\n");
  struct { size_t sz; const char* what; } cases[] = {
    {10240,        "decode, 1 tok  (n_embd 5120 bf16)"},
    {10240*8,      "decode, batch 8"},
    {10240*32,     "decode, batch 32"},
    {10240*512,    "prefill chunk 512"},
    {10240*2048,   "prefill chunk 2048"},
  };
  for(auto&c : cases){
    int it = c.sz < (1<<20) ? 2000 : 100;
    hipMemcpyPeer(pb,1,pa,0,c.sz); CK(hipDeviceSynchronize());
    auto t0=clk::now();
    for(int i=0;i<it;i++) hipMemcpyPeer(pb,1,pa,0,c.sz);
    CK(hipDeviceSynchronize());
    auto t1=clk::now();
    double per = us(t0,t1)/it;
    printf("%9zu B  %8.1f us  %7.2f GB/s   %s\n",
           c.sz, per, c.sz/(per*1e-6)/1e9, c.what);
  }

  // the money number: one token's worth of TP=2 comm for 27B dense
  printf("\n-- simulated TP=2 all-reduce chain, Qwen3.8-27B (65 layers x 2) --\n");
  for(int reps : {1,10}){
    auto t0=clk::now();
    for(int r=0;r<reps;r++)
      for(int i=0;i<130;i++){
        hipMemcpyPeer(pb,1,pa,0,10240);
        hipMemcpyPeer(pa,0,pb,1,10240);
      }
    CK(hipDeviceSynchronize());
    auto t1=clk::now();
    double per_tok = us(t0,t1)/reps/1000.0;
    printf("  %.3f ms of comm per token  ->  %.0f tok/s ceiling from comm alone\n",
           per_tok, 1000.0/per_tok);
  }
  return 0;
}
