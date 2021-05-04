//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
#include <stdio.h>
#include <stdlib.h> 
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "info.hpp"


#define DEFAULT_PLATFORM 0
#define NUM_BUFFER_ELEMENTS 16
#define UNIQUE_ARGS 5

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create all kernels
//
std::vector<cl_kernel> CreateKernels(cl_program program)
{
    cl_int errNum;
    std::vector<cl_kernel> kernels

    for (int i = 0; i < UNIQUE_ARGS; i++)
    {
        if (i == 0):
        {
            cl_kernel kernel = clCreateKernel(program, "square", &errNum);
        }
        else if (i == 1):
        {
            cl_kernel kernel = clCreateKernel(program, "cube", &errNum);
        }
        else if (i == 1):
        {
            cl_kernel kernel = clCreateKernel(program, "identity", &errNum);
        }
        else if (i == 1):
        {
            cl_kernel kernel = clCreateKernel(program, "double", &errNum);
        }
        else:
        {
            cl_kernel kernel = clCreateKernel(program, "triple", &errNum);
        }
        kernels.push_back(kernel);
        checkErr(errNum, "clCreateKernel");
    }

    return kernels;
}

///
//  Create memory objects used as the arguments to the kernel
//
void CreateBuffers(cl_context context, cl_mem memObjects[5])
    cl_int errNum;
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * NUM_BUFFER_ELEMENTS, 
                                    NULL, &errNum);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * NUM_BUFFER_ELEMENTS, 
                                    NULL, &errNum);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * NUM_BUFFER_ELEMENTS, 
                                    NULL, &errNum);
    memObjects[3] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * NUM_BUFFER_ELEMENTS, 
                                    NULL, &errNum);
    memObjects[4] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * NUM_BUFFER_ELEMENTS, 
                                    NULL, &errNum);                                                                                                           

    checkErr(errNum, "clCreateBuffer");                 
}

///
// Set up kernels, queue events, and read back result to host 
// according to command line input
//
void setupAndExecute(std::vector<cl_kernel> &kernels, cl_mem buffers[5], 
                     int arg, cl_context context, cl_command_queue command_queue,
                     int *inputOutput)
{
    cl_int errNum;
    // Write input data to each buffer
    for (int i = 0; i < UNIQUE_ARGS; i++)
    {
        errNum = clEnqueueWriteBuffer(command_queue, buffers[i], CL_TRUE, 0,
                                      sizeof(int) * NUM_BUFFER_ELEMENTS, 
                                      (void*)inputOutput, 0, NULL, NULL);
    }

    std::vector<cl_event> events;
    size_t gWI = NUM_BUFFER_ELEMENTS;
    // Set kernel args and enqueue
    for (int i = 0; i < UNIQUE_ARGS; i++)
    {
        cl_event event;
        errNum = clSetKernelArg(kernels[i], 0, sizeof(cl_mem), (void *)&buffers[i]);
        checkErr(errNum, "clSetKernelArg");
        clEnqueueNDRangeKernel(command_queue, kernels[i], 1, NULL, 
                               (const size_t*)&gWI, (const size_t*)NULL, 0, 0, 
                               &event0);
        events.push_back(event);
    }
    // Wait for all events to complete
 	clWaitForEvents(UNIQUE_ARGS, &events[0]);
    // Read back the event according to input arg
    clEnqueueReadBuffer(command_queue, buffers[arg], CL_TRUE, 0, 
                        sizeof(int) * NUM_BUFFER_ELEMENTS,
                        (void*)inputOutput, 0, NULL, NULL);

}

///
//	main() for OpenCL event assignment. Outputs
//  different events to host based on input arg
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_context context = 0;
    cl_program program = 0;
    cl_command_queue command_queue = 0;
    cl_device_id device = 0;
    std::vector<cl_kernel> kernels;
    cl_mem buffers[5] = {0, 0, 0, 0, 0};
    int *inputOutput;

    // Read command line argument
    int arg;
	if (argc >= 2) {
		arg = atoi(argv[1]);
	}

    // Create an OpenCL context on first available platform
    context = CreateContext();
    // Create OpenCL program from openCL_math.cl kernel source
    program = CreateProgram(context, device, "events.cl");
    // Create a command-queue on the first device available
    // on the created context
    command_queue = CreateCommandQueue(context, &device);
    // Create all kernels
    kernels = CreateKernels(program);

    // Populate host array
    inputOutput = new int[NUM_BUFFER_ELEMENTS];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    {
        inputOutput[i] = i;
    }

    // Create buffers for each event
    CreateBuffers(context, buffers);

    // start timing
    clock_t start = clock();
    // Execute kernels and read data back to host as specified by input arg
    setupAndExecute(kernels, buffers[5], arg, context, command_queue, inputOutput);
    // end timing
    clock_t end = clock();
    double elapsed = double(end - start)/CLOCKS_PER_SEC;
    std::cout << "Execution time: " << elapsed << "\n" << std::endl;

    // Display output in rows
    for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++)
    {
     std::cout << " " << inputOutput[elems];
    }
    std::cout << std::endl;

    return 0;
}
