// average.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage


__kernel void avg(
    const __global uint * const filter,
    __global float * const output,
    const int filterWidth
    )
{
    float sum = 0;
    for (int i = 0; i < 2*filterWidth; i++)
    {
        sum  += filter[i];
    }
    output[0] = sum/(2.0 * filterWidth);
}