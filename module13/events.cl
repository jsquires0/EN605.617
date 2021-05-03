// simple.cl
//
//    This is a simple example demonstrating multiple event usage

__kernel void square(__global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] * buffer[id];
}

__kernel void cube(__global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] * buffer[id] * buffer[id];
}

__kernel void identity(__global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id];
}

__kernel void double(__global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] + buffer[id];
}

__kernel void triple(__global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] + buffer[id] + buffer[id];
}