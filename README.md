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