# dqn-hfo

This is an continuous action deep reinforcement learning agent for the
RoboCup 2D domain. The domain can be found and downloaded from
https://github.com/mhauskn/HFO.

This repo is designed to work with the latest version of
[Caffe](https://github.com/BVLC/caffe) (currently commit
ff16f6e43dd718921e5203f640dd57c68f01cdb3) with the following minor
changes:

```
--- a/include/caffe/solver.hpp
+++ b/include/caffe/solver.hpp
@@ -67,6 +67,7 @@ class Solver {
     return test_nets_;
   }
   int iter() { return iter_; }
+  void set_iter(int new_iter) { iter_ = new_iter; }
 
   // Invoked at specific points during an iteration
   class Callback {
@@ -84,7 +85,6 @@ class Solver {
 
   void CheckSnapshotWritePermissions();
 
- protected:
   // Make and apply the update value for the current iteration.
   virtual void ApplyUpdate() = 0;
   // The Solver::Snapshot function implements the basic snapshotting utility
@@ -95,6 +95,7 @@ class Solver {
   string SnapshotFilename(const string extension);
   string SnapshotToBinaryProto();
   string SnapshotToHDF5();
+ protected:
   // The test routine
   void TestAll();
   void Test(const int test_net_id = 0);
```

## Installation

1. First install the correct version of Caffe:
  1. ```git clone https://github.com/BVLC/caffe.git```
  2. ```cd caffe && git checkout ff16f6e43dd718921e5203f640dd57c68f01cdb3```
  3. Apply the changes to solver listed above
  4. Follow installation instructions at https://github.com/BVLC/caffe
2. Next install HFO:
  1. ```git clone https://github.com/LARG/HFO.git```
  2. Follow installation instructions at https://github.com/LARG/HFO
3. Now we are ready to install dqn-hfo:
  1. ```git clone https://github.com/mhauskn/dqn-hfo.git```
  2. ```cd dqn-hfo```
  3. ```cmake -DCMAKE_BUILD_TYPE=Release -DCAFFE_ROOT_DIR=/u/mhauskn/projects/caffe/ -DHFO_ROOT_DIR=/u/mhauskn/projects/HFO/ .``` You will have to change the paths to point to your installation of caffe and HFO
  4. ```make -j4```
4. Run a test job: ```mkdir state && ./dqn -save state/test -alsologtostderr```

## Errors

1. Cannot find cublas_v2.h:
```device_alternate.hpp:34:23: fatal error: cublas_v2.h: No such file or directory
 #include <cublas_v2.h>
                       ^
compilation terminated.```

Solution: Include your Cuda path in the installation:

  1. ```locate cublas_v2.h``` -- this should give you the path to your cuda installation
  2. ```export CPLUS_INCLUDE_PATH=/your/cuda/path:$CPLUS_INCLUDE_PATH```

2. ```caffe/include/caffe/blob.hpp:9:34: fatal error: caffe/proto/caffe.pb.h: No such file or directory
 #include "caffe/proto/caffe.pb.h"```

Solution: Symlink the built proto files.
  1. ```cd your_caffe_dir/include/caffe```
  2. ```ln -s ../../.build_release/src/caffe/proto/ .```

## Citing

If this repository has helped your research, please cite the following:

    @InProceedings{ICLR16-hausknecht,
      author = {Matthew Hausknecht and Peter Stone},
      title = {Deep Reinforcement Learning in Parameterized Action Space},
      booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
      location = {San Juan, Puerto Rico},
      month = {May},
      year = {2016},
    }