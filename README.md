# Parallelized Hausdorff-Based Matching
## APMA2822B - Introduction to Parallel Computing on Heterogeneous(CPU+GPU) Systems 
### Final Project (*May, 2019*)


#### Yang Jiao yang_jiao@brown.edu

#### Geng Yang   geng_yang@brown.edu

## Software Architecture

![image](https://github.com/Genggg/ParaHausdorff/blob/master/flow.png)

### Dependencies
- OpenCV 3.4.1 (only for loading images)
- OpenMP 4.0
- CUDA 10.0.130
### Build System
- Redhat 7 (Brown's CCV)


## Instructions for compiling and running
This software can be built on Linux, macOS and Windows, the instructions are as follows.
1. Clone the repository.
```
$ git@github.com:Genggg/ParaHausdorff.git
```
2. If build on CCV, please load the following modules.
```
$ moudle load opencv/3.4.1
$ moudle load cuda/10.0.130
```
3. Generate MakeFile and Compile.

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
4. Modify the execution command in ```cuda.sh```, in which the following args are the paths of target image and template respectively.
```
./main ../coins.jpg ../coin_T.png
```
5. Submit the task and check the result
```
$ sbatch cuda.sh
```
