#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <string.h>
#include <iostream>
#include <helper_string.h>
#include <cuda_runtime.h>
#include <npp.h>

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}
/*
 * This function uses NPP to mirror an input image along the vertical axis.
 */
int main(int argc, char *argv[])
{   // Find input image filepath
    std::string sFilename;
    sFilename = sdkFindFilePath("Lena.pgm", argv[0]);
    std::string sResultFilename = sFilename;
    // Set output image filepath
    std::string::size_type dot = sResultFilename.rfind('.');
    if (dot != std::string::npos)
        {sResultFilename = sResultFilename.substr(0, dot);}
    sResultFilename += "_Mirrored.pgm";

    // declare 8-bit single channel host image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load image
    npp::loadImage(sFilename, oHostSrc);
    // timing
    cudaEvent_t start_time = get_time();
    // transfer host -> device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    // create struct with ROI size
    NppiSize oROI = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oROI.width, oROI.height);
    // Flip the input image along the vertical axis
    nppiMirror_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), 
                    oDeviceDst.data(), oDeviceDst.pitch(), 
                    oROI, NPP_VERTICAL_AXIS);
    // transfer result device -> host
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    // end timing
    cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);
    printf("NPP Mirror ran in %3.3f ms\n", delta);

    return EXIT_SUCCESS;
}

