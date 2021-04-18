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
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Create array of predefined values
#define NUM_BUFFER_ELEMENTS 16
cl_int inputHost[NUM_BUFFER_ELEMENTS] = 
{
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 
    10, 11, 12, 13, 14, 15, 16}
};

cl_float output[NUM_BUFFER_ELEMENTS];

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



/*///
//  Create memory objects used as the arguments to the kernel
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[2],
                      int *input, int use_pinned)
{
    if (use_pinned)
    {
        memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | 
                                       CL_MEM_ALLOC_HOST_PTR,
                                       sizeof(int) * NUM_BUFFER_ELEMENTS,
                                       input, NULL);
        // Create sub-buffer
        cl_buffer_region region = 
        {
            2 * sizeof(int), 
            2 * sizeof(int)
        };
        memObjects[1] = clCreateSubBuffer(memObjects[0], CL_MEM_READ_WRITE | 
                                       CL_MEM_ALLOC_HOST_PTR,
                                       CL_BUFFER_CREATE_TYPE_REGION,
                                       &region, NULL);
    }
    else
    {
        memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | 
                                       CL_MEM_COPY_HOST_PTR,
                                       sizeof(int) * NUM_BUFFER_ELEMENTS,
                                       input, NULL);

        // Create sub-buffer
        cl_buffer_region region = 
        {
            2 * sizeof(int), 
            4 * sizeof(int)
        };
        memObjects[1] = clCreateSubBuffer(memObjects[0], CL_MEM_READ_WRITE | 
                                       CL_MEM_COPY_HOST_PTR,
                                       CL_BUFFER_CREATE_TYPE_REGION,
                                       &region, NULL);
    }
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * NUM_BUFFER_ELEMENTS, NULL, NULL);
                                             
    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL  )
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}*/

///
//  Cleanup any created OpenCL resources
//
/*void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0) clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}*/

/*
 * Queues a kernel and reads the result back to host
 */
/*int execute_kernel(cl_command_queue commandQueue, cl_kernel kernel, 
                   size_t globalWorkSize[1], size_t localWorkSize[1], 
                   cl_int errNum, cl_mem memObjects[7], float *result)
{
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                globalWorkSize, localWorkSize,
                                0, NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
    }
    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
    }

    return errNum;

}*/
/*
 * Sets up kernel args and executes on device
 */
/*int setupAndExecuteMath(cl_kernel kernel, cl_mem memObjects[3],
                        cl_command_queue commandQueue, float *output){

    // Set the kernel arguments 
    cl_int errNum;
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);

    if (errNum != CL_SUCCESS)
    {
        printf("%d", errNum);
        std::cerr << "Error setting kernel arguments." << std::endl;
    }

    size_t globalWorkSize[1] = { NUM_BUFFER_ELEMENTS };
    size_t localWorkSize[1] = { 1 };

    execute_kernel(commandQueue, kernel, globalWorkSize, localWorkSize, 
                   errNum, memObjects, 2, output);

    return errNum;


}*/
/*
 * Print vector math output to a file
 */
/*void printResults(float *addOut, float *subOut, float *multOut, float *modOut, 
                  float *powOut)
{
    // Print to file
    FILE * outFile;
    outFile = fopen("computed_arrays.txt","w");
    for (int i=0; i<ARRAY_SIZE; i++)
    {
        fprintf(outFile, "%f\t %f\t %f\t %f\t %f\t \n", 
                addOut[i], subOut[i], multOut[i], modOut[i], powOut[i]);
    }             
}*/
///
//	main() for openCL Math example. Computes add, sub, mult, mod, and pow
//  on two arrays using either pinned or pageable host memory
//
int memTest(int use_pinned)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel;
    cl_mem memObjects[3] = {0, 0, 0};
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    // Create OpenCL program from average.cl kernel source
    program = CreateProgram(context, device, "average.cl");

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float averaged[NUM_BUFFER_ELEMENTS];

    //if (!CreateMemObjects(context, memObjects, a, b, use_pinned)){ return 1;}
    //INSERT HERE

    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | 
                                       CL_MEM_COPY_HOST_PTR,
                                       sizeof(int) * NUM_BUFFER_ELEMENTS,
                                       input, NULL);
    // Create sub-buffer
    size_t origin[3] = {0*sizeof(int), 0, 0};
    size_t size[3] = {2* sizeof(int), 2, 1};
    cl_buffer_region region = 
    {
        origin[3], 
        size[3]
    };
    memObjects[1] = clCreateSubBuffer(memObjects[0], CL_MEM_READ_WRITE | 
                                    CL_MEM_ALLOC_HOST_PTR,
                                    CL_BUFFER_CREATE_TYPE_REGION,
                                    &region, NULL);







    // Create OpenCL kernels
    /*kernel = clCreateKernel(program, "average", NULL);

    // start timing
    clock_t start = clock();
   
    // Set the kernel arguments (result, a, b)
    setupAndExecuteMath(kernel, memObjects, 2, commandQueue, output);

    
    // end timing
    clock_t end = clock();
    double elapsed = double(end - start)/CLOCKS_PER_SEC;
    use_pinned ? std::cout << "Pinned execution time: " << elapsed << "\n" << std::endl
               : std::cout << "Pageable execution time: " << elapsed << "\n" << std::endl;

    // Print to file
    //printResults(addOut, subOut, multOut, modOut, powOut);

    Cleanup(context, commandQueue, program, add_kernel, sub_kernel, 
            mult_kernel, mod_kernel, pow_kernel, memObjects);
    */
    return 0;
}

/*
 *	Calls test function once using pageable and once with pinned host memory for
 *  two sets of input data sizes.
 */
int main(int argc, char** argv)
{
    int use_pinned = 1;
    memTest(!use_pinned);

}

