// average.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void average(
    __global uint * filter,
    __global float * const output,
    )
{
	int gid = get_global_id(0);
    float sum = 0;
    for (int r = 0; r < 4; r++)
    {
        sum += filter[gid + r];
    }
	output[gid] = sum / 4.0;
}