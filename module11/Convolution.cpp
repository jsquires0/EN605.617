//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

cl_float mask[maskHeight][maskWidth] =
{
	{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
	{0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25},
	{0.25, 0.50, 0.75, 0.75, 0.75, 0.55, 0.25},
	{0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25},
	{0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25},
	{0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25},
	{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
};

const unsigned int outputSignalWidth  = inputSignalWidth - maskWidth + 1;
const unsigned int outputSignalHeight = inputSignalHeight - maskHeight + 1;

cl_float outputSignal[outputSignalHeight][outputSignalWidth];



/*
 * Function to create random 49 x 49 signal
 */
void rand_signal(cl_uint signal[inputSignalHeight][inputSignalWidth])
{
	for (int i=0; i< inputSignalHeight; i++){
		for (int j=0; j < inputSignalWidth; j++){
			signal[i][j] = rand() % 9;
		}
	}
}

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

///
//  Create memory objects used as the arguments to the kernel
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      cl_uint inputSignal[inputSignalHeight][inputSignalWidth], 
					  cl_float mask[maskHeight][maskWidth], 
					  int use_pinned, cl_int errNum)
{
	cl_mem inputSignalBuffer;
	cl_mem maskBuffer;
	cl_mem outputSignalBuffer;

	// Pageable vs Pinned
	cl_mem_flags hostMemType;
	use_pinned ? hostMemType = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR
			   : hostMemType = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(context, hostMemType, 
			sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
			static_cast<void *>(inputSignal), &errNum);

	maskBuffer = clCreateBuffer(context, hostMemType,
		sizeof(cl_float) * maskHeight * maskWidth,
		static_cast<void *>(mask),&errNum);

	outputSignalBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			sizeof(cl_float) * outputSignalHeight * outputSignalWidth, 
			NULL, &errNum);

	memObjects[0] = inputSignalBuffer;
	memObjects[1] = maskBuffer;
	memObjects[2] = outputSignalBuffer;
                                             
    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}
/*
 * Print convolution output to file
 */
void printResults(cl_float outputSignal[outputSignalHeight][outputSignalWidth])
{
    // Print to file
    FILE * outFile;
    outFile = fopen("computed_array.txt","w+");
    for (int y = 0; y < outputSignalHeight; y++)
	{
		for (int x = 0; x < outputSignalWidth; x++)
		{
			fprintf(outFile, "%f ", outputSignal[y][x]);
		}
		fprintf(outFile, "\n");
	}
         
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, 
             cl_mem memObjects[3])
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

}


///
//	main() for Convolution example
//
int memTest(int use_pinned, 
		    cl_uint inputSignal[inputSignalHeight][inputSignalWidth])
{
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem memObjects[3] = {0, 0, 0};

    // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");
	
	// Pageable vs Pinned
	cl_mem_flags hostMemType;
	use_pinned ? hostMemType = CL_MEM_ALLOC_HOST_PTR
			   : hostMemType = CL_MEM_COPY_HOST_PTR;

	// start timing
    clock_t start = clock();
	if (!CreateMemObjects(context, memObjects, inputSignal, mask, use_pinned, errNum)){ return 1;}

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
    const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueNDRangeKernel");
    
	errNum = clEnqueueReadBuffer(
		queue, 
		memObjects[2], 
		CL_TRUE,
        0, 
		sizeof(cl_float) * outputSignalHeight * outputSignalHeight, 
		outputSignal,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

	 // end timing
    clock_t end = clock();
    double elapsed = double(end - start)/CLOCKS_PER_SEC;

    use_pinned ? std::cout << "Pinned execution time: " << elapsed << "\n" << std::endl
               : std::cout << "Pageable execution time: " << elapsed << "\n" << std::endl;

    // Output the result buffer to file and cleanup
    printResults(outputSignal);
	Cleanup(context, queue, program, kernel, memObjects);
    std::cout << std::endl << "Executed program succesfully." << std::endl;
	
	return 0;
}

/*
 *	Calls test function once using pageable and once with pinned host memory for
 *  two separate input signals
 */
int main(int argc, char** argv)
{
    int use_pinned = 1;
	cl_uint inputSignal[inputSignalHeight][inputSignalWidth];

	// Test first signal
	rand_signal(inputSignal);
    memTest(use_pinned, inputSignal);
    memTest(!use_pinned, inputSignal);

	// Test second signal
	rand_signal(inputSignal);
    memTest(use_pinned, inputSignal);
    memTest(!use_pinned, inputSignal);
}

