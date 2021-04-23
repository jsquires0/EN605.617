//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// openCL_math.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Create array of predefined values
#define NUM_BUFFER_ELEMENTS 16
// Define sizes of filter (2x2)
#define FILTER_ELEMENTS 4
// Number of filters 
int N_SUB_BUFFS = NUM_BUFFER_ELEMENTS - FILTER_ELEMENTS + 1;
// total number of sub buffers (one filter, one output) + output/input buffers
int TOTAL_N_BUFFERS = N_SUB_BUFFS * 2 + 2;

cl_int inputHost[NUM_BUFFER_ELEMENTS] = 
{
	1, 2, 3, 4, 5, 6, 7, 8, 9, 
    10, 11, 12, 13, 14, 15, 16
};

// Create array to hold averaged output values
cl_float averaged[NUM_BUFFER_ELEMENTS] = 
{
	0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0
};

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
//	main() for sub-buffer assignment. Computes average of input array in 
//  4 element moving window using sub buffers.
//
int memTest(int use_pinned, cl_context context, cl_command_queue commandQueue,
cl_program program, cl_device_id device, cl_kernel kernel, cl_int errNum)
{
    /*cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel;
    cl_mem memObjects[TOTAL_N_BUFFERS];
    cl_int errNum;*/
    cl_mem memObjects[TOTAL_N_BUFFERS];

    /*// Create an OpenCL context on first available platform
    context = CreateContext();
    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    // Create OpenCL program from average.cl kernel source
    program = CreateProgram(context, device, "average.cl");

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "avg", &errNum);
	checkErr(errNum, "clCreateKernel err");*/

    // Create input and output buffers
    // Pageable vs Pinned
	cl_mem_flags hostMemType;
	use_pinned ? hostMemType = CL_MEM_ALLOC_HOST_PTR 
			   : hostMemType = CL_MEM_COPY_HOST_PTR;
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | 
                                       hostMemType,
                                       sizeof(int) * NUM_BUFFER_ELEMENTS,
                                       inputHost, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * N_SUB_BUFFS,
                                       averaged, NULL);

    // start timing
    clock_t start = clock();
    
    // Create sub-buffer filters, call kernel
    std::vector<cl_event> events;
    int FILTER_WIDTH = FILTER_ELEMENTS / 2;
    for (int i = 0; i < N_SUB_BUFFS; i++) 
    {
        cl_event event;
        cl_buffer_region inputRegion = 
        {
            //specify origin, size
            i * FILTER_ELEMENTS * sizeof(int),
            FILTER_ELEMENTS * sizeof(int)
        };
        // after averaging, output is float
        cl_buffer_region outputRegion = 
        {
            //specify origin, size
            i * sizeof(float),
            sizeof(float)
        };
        memObjects[i+2] = clCreateSubBuffer(memObjects[0], CL_MEM_READ_WRITE |
                                CL_MEM_COPY_HOST_PTR,
                                CL_BUFFER_CREATE_TYPE_REGION,
                                &inputRegion, NULL);
        memObjects[i+3] = clCreateSubBuffer(memObjects[1], CL_MEM_READ_WRITE,
                                CL_BUFFER_CREATE_TYPE_REGION,
                                &outputRegion, NULL);
        
        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memObjects[i+2]);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memObjects[i+3]);
        errNum |= clSetKernelArg(kernel, 2, sizeof(int), &FILTER_WIDTH);
        size_t globalWorkSize[1] = {1};
        size_t localWorkSize[1] = {1};

        errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                globalWorkSize, localWorkSize,
                                0, NULL, &event);
        events.push_back(event);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error queuing kernel for execution." << std::endl;
        }
    }

    clWaitForEvents(N_SUB_BUFFS, &events[0]);

    // Read back computed data
    clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE,
        0, sizeof(float) * N_SUB_BUFFS, averaged,
        0, NULL, NULL);
    
    // end timing
    clock_t end = clock();
    double elapsed = double(end - start)/CLOCKS_PER_SEC;
    use_pinned ? std::cout << "Pinned execution time: " << elapsed << "\n" << std::endl
               : std::cout << "Pageable execution time: " << elapsed << "\n" << std::endl;
    return 0;
}

/*
 *	Calls test function once using pageable and once with pinned host memory for
 *  two sets of input data sizes. 
 */
int main(int argc, char** argv)
{

    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel;
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    // Create OpenCL program from average.cl kernel source
    program = CreateProgram(context, device, "average.cl");

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "avg", &errNum);
	checkErr(errNum, "clCreateKernel err");


    int use_pinned = 1;
    memTest(!use_pinned, context, commandQueue, program, device, kernel, errNum);
    memTest(use_pinned, context, commandQueue, program, device, kernel, errNum);

}