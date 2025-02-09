#include "naive_cuda_simulation.cuh"
#include "physics/gravitation.h"
#include "physics/mechanics.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_wrappers.cuh"

void NaiveCudaSimulation::allocate_device_memory(Universe& universe, void** d_weights, void** d_forces, void** d_velocities, void** d_positions){
    parprog_cudaMalloc(d_weights, universe.num_bodies * sizeof(double));
    parprog_cudaMalloc(d_forces, universe.num_bodies * sizeof(double2));
    parprog_cudaMalloc(d_velocities, universe.num_bodies * sizeof(double2));
    parprog_cudaMalloc(d_positions, universe.num_bodies * sizeof(double2));
}

void NaiveCudaSimulation::free_device_memory(void** d_weights, void** d_forces, void** d_velocities, void** d_positions){
    parprog_cudaFree(d_weights);
    d_weights = nullptr;
    parprog_cudaFree(d_forces);
    d_forces = nullptr;
    parprog_cudaFree(d_velocities);
    d_velocities = nullptr;
    parprog_cudaFree(d_positions);
    d_positions = nullptr;



}

void NaiveCudaSimulation::copy_data_to_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    parprog_cudaMemcpy(d_weights, universe.weights.data(), universe.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
    std::vector<double2> converted_forces;
    //Converting vec2d to double2
    for (int i = 0; i < universe.num_bodies; i++) {
        double2 conv;
        conv.x = universe.forces[i][0];
        conv.y = universe.forces[i][1];
        converted_forces.push_back(conv);
    }
    parprog_cudaMemcpy(d_forces, converted_forces.data(), universe.num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    std::vector<double2> converted_velocities;
    //Converting vec2d to double2
    for (int i = 0; i < universe.num_bodies; i++) {
        double2 conv;
        conv.x = universe.velocities[i][0];
        conv.y = universe.velocities[i][1];
        converted_velocities.push_back(conv);
    }
    parprog_cudaMemcpy(d_velocities, converted_velocities.data(), universe.num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    std::vector<double2> converted_positions;
    //Converting vec2d to double2
    for (int i = 0; i < universe.num_bodies; i++) {
        double2 conv;
        conv.x = universe.positions[i][0];
        conv.y = universe.positions[i][1];
        converted_positions.push_back(conv);
    }
    parprog_cudaMemcpy(d_positions, converted_positions.data(), universe.num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
}

void NaiveCudaSimulation::copy_data_from_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions) {
    parprog_cudaMemcpy(universe.weights.data(), d_weights, universe.num_bodies * sizeof(double), cudaMemcpyDeviceToHost);
    std::vector<double2> received_forces(universe.num_bodies);
    parprog_cudaMemcpy(received_forces.data(), d_forces, universe.num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    //Converting double2 to vec2d
    std::vector<Vector2d<double>> converted_forces;
    for (int i = 0; i < universe.num_bodies; i++) {
        Vector2d<double> conv(received_forces[i].x, received_forces[i].y);
        converted_forces.push_back(conv);
    };
    universe.forces = converted_forces;

    std::vector<double2> received_velocities(universe.num_bodies);
    parprog_cudaMemcpy(received_velocities.data(), d_velocities, universe.num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    //Converting double2 to vec2d 
    std::vector<Vector2d<double>> converted_velocities; 
    for (int i = 0; i < universe.num_bodies; i++) { 
        Vector2d<double> conv(received_velocities[i].x, received_velocities[i].y);
        converted_velocities.push_back(conv); 
    };
    universe.velocities = converted_velocities;

    std::vector<double2> received_positions(universe.num_bodies);
    parprog_cudaMemcpy(received_positions.data(), d_positions, universe.num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    //Converting double2 to vec2d
    std::vector<Vector2d<double>> converted_positions;
    for (int i = 0; i < universe.num_bodies; i++) {
        Vector2d<double> conv(received_positions[i].x, received_positions[i].y);
        converted_positions.push_back(conv);
    };
    universe.positions = converted_positions;
}

__global__
void calculate_forces_kernel(std::uint32_t num_bodies, double2* d_positions, double* d_weights, double2* d_forces){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies)
        return;

    double2 position_i = d_positions[i];
    double weight_i = d_weights[i];

    double2 force = { 0.0, 0.0 };

    
    for (std::uint32_t j = 0; j < num_bodies; ++j) {
        if (j == i)
            continue;  

        double2 position_j = d_positions[j];

        
        double2 direction;
        direction.x = position_j.x - position_i.x;
        direction.y = position_j.y - position_i.y;

        
        double distance = sqrt(pow(direction.x, 2) + pow(direction.y, 2));
        
    	double f = 6.67430e-11 * weight_i * d_weights[j] / (distance * distance);
            
    	double force_factor = f / distance;
    	force.x += direction.x * force_factor;
    	force.y += direction.y * force_factor;
    }
    d_forces[i] = force;
}

void NaiveCudaSimulation::calculate_forces(Universe& universe, void* d_positions, void* d_weights, void* d_forces){
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;
    if (universe.num_bodies % block_dim == 0) {
        grid_dim = universe.num_bodies / block_dim;
    }
    else {
        grid_dim = (universe.num_bodies - (universe.num_bodies % block_dim) + block_dim) / block_dim;
    }

    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_forces_kernel<<<gridDim, blockDim>>>(universe.num_bodies, (double2*) d_positions, (double*) d_weights, (double2*) d_forces);

}

__global__
void calculate_velocities_kernel(std::uint32_t num_bodies, double2* d_forces, double* d_weights, double2* d_velocities){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies)
        return;

    double2 velocity;

    double2 force = d_forces[i];
    double weight = d_weights[i];
    double2 base_velocity = d_velocities[i];

    double2 acceleration;
    acceleration.x = force.x / weight;
	acceleration.y = force.y / weight;

    velocity.x = base_velocity.x + (acceleration.x * 2.628e+6);
    velocity.y = base_velocity.y + (acceleration.y * 2.628e+6);

    d_velocities[i] = velocity;
}

void NaiveCudaSimulation::calculate_velocities(Universe& universe, void* d_forces, void* d_weights, void* d_velocities){
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;
    if (universe.num_bodies % block_dim == 0) {
        grid_dim = universe.num_bodies / block_dim;
    }
    else {
        grid_dim = (universe.num_bodies - (universe.num_bodies % block_dim) + block_dim) / block_dim;
    }

    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);

	calculate_velocities_kernel << <gridDim, blockDim >> > (universe.num_bodies, (double2*)d_forces, (double*)d_weights, (double2*)d_velocities);
}

__global__
void calculate_positions_kernel(std::uint32_t num_bodies, double2* d_velocities, double2* d_positions){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies)
        return;

    double2 new_position;

    double2 velocity = d_velocities[i];
    double2 position = d_positions[i];

    double2 movement;
    movement.x = velocity.x * 2.628e+6;
    movement.y = velocity.y * 2.628e+6;

    new_position.x = position.x + movement.x;
    new_position.y = position.y + movement.y;

    d_positions[i] = new_position;

}

void NaiveCudaSimulation::calculate_positions(Universe& universe, void* d_velocities, void* d_positions){
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;
    if (universe.num_bodies % block_dim == 0) {
        grid_dim = universe.num_bodies / block_dim;
    }
    else {
        grid_dim = (universe.num_bodies - (universe.num_bodies % block_dim) + block_dim) / block_dim;
    }

	dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_positions_kernel << <gridDim, blockDim >> > (universe.num_bodies, (double2*)d_velocities, (double2*)d_positions);
}

void NaiveCudaSimulation::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs) {
    //init pointers
    void* d_weights;
    void* d_forces;
    void* d_velocities;
    void* d_positions;
    //copy data to device
    allocate_device_memory(universe, &d_weights, &d_forces, &d_velocities, &d_positions);
    copy_data_to_device(universe, &d_weights, &d_forces, &d_velocities, &d_positions);
    for (int i = 0; i < num_epochs; i++) {
        //Calculations (num_epochs times)
        calculate_forces(universe, d_positions, d_weights, d_forces);
        calculate_velocities(universe, d_forces, d_weights, d_velocities);
        calculate_positions(universe, d_velocities, d_positions);
        universe.current_simulation_epoch++;

        //plotcheck, copied from naive-sequential
        if (create_intermediate_plots) {
            if ((universe.current_simulation_epoch % plot_intermediate_epochs) == 0) {
                std::vector<std::uint8_t> pixels = get_pixels(plotter.get_plot_width(), plotter.get_plot_height(), plotter.get_plot_bounding_box(), d_positions, universe.num_bodies); 
                plotter.add_active_pixels_to_image(pixels);
                plotter.write_and_clear();
            }
        }
    }
    //Copy results to universe
    copy_data_from_device(universe, d_weights, d_forces, d_velocities, d_positions); 
    //free space
    free_device_memory(&d_weights, &d_forces, &d_velocities, &d_positions);
}      

__global__
void get_pixels_kernel(std::uint32_t num_bodies, double2* d_positions, std::uint8_t* d_pixels, std::uint32_t plot_width, std::uint32_t plot_height, double plot_bounding_box_x_min, double plot_bounding_box_x_max, double plot_bounding_box_y_min, double plot_bounding_box_y_max){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies)
        return;
    double2 position = d_positions[i];
    if (plot_bounding_box_x_min < position.x && position.x < plot_bounding_box_x_max && plot_bounding_box_y_min < position.y && position.y < plot_bounding_box_y_max) {
        std::uint32_t x = ((position.x - plot_bounding_box_x_min) / (plot_bounding_box_x_max - plot_bounding_box_x_min))*(plot_width-1);
        std::uint32_t y = ((position.y - plot_bounding_box_y_min) / (plot_bounding_box_y_max - plot_bounding_box_y_min))*(plot_height-1);
        d_pixels[y * plot_width + x] = 1;
    }

}

std::vector<std::uint8_t> NaiveCudaSimulation::get_pixels(std::uint32_t plot_width, std::uint32_t plot_height, BoundingBox plot_bounding_box, void* d_positions, std::uint32_t num_bodies){
    std::uint32_t num_of_pixels = plot_height * plot_width;
    std::vector<std::uint8_t> pixels(num_of_pixels, 0);
    void* d_pixels;
    parprog_cudaMalloc(&d_pixels, num_of_pixels * sizeof(std::uint8_t));
    parprog_cudaMemcpy(d_pixels, &pixels, num_of_pixels * sizeof(std::uint8_t), cudaMemcpyHostToDevice); //Might be unecessary
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;
    if (num_bodies % block_dim == 0) {
        grid_dim = num_bodies / block_dim;
    }
    else {
        grid_dim = (num_bodies - (num_bodies % block_dim) + block_dim) / block_dim;
    }
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    get_pixels_kernel << <gridDim, blockDim >> > (num_bodies, d_positions, d_pixels, plot_width, plot_height, plot_bounding_box.x_min, plot_bounding_box.x_max, plot_bounding_box.y_min, plot_bounding_box.y_max);
    parprog_cudaMemcpy(&pixels, d_pixels, num_of_pixels * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    parprog_cudaFree(d_pixels);
    return pixels;
}

__global__
void compress_pixels_kernel(std::uint32_t num_raw_pixels, std::uint8_t* d_raw_pixels, std::uint8_t* d_compressed_pixels){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i*8 >= num_raw_pixels)
        return;
    std::uint8_t compressed_pixel = 0;
    for (int k =0; k<8; k++){
        if (i * 8 + k < num_raw_pixels && d_raw_pixels[i * 8 + k] != 0) {
            compressed_pixel += 1 << k;
        }
    }
    d_compressed_pixels[i] = compressed_pixel;
}

void NaiveCudaSimulation::compress_pixels(std::vector<std::uint8_t>& raw_pixels, std::vector<std::uint8_t>& compressed_pixels){
    std::uint32_t num_raw_pixels = raw_pixels.size();
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;
    if (num_raw_pixels % block_dim == 0) {
        grid_dim = num_raw_pixels / block_dim;
    }
    else {
        grid_dim = (num_raw_pixels - (num_raw_pixels % block_dim) + block_dim) / block_dim;
    }
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);

    void* d_raw_pixels;
    void* d_compressed_pixels;
    parprog_cudaMalloc(&d_raw_pixels, num_raw_pixels * sizeof(uint8_t));
    parprog_cudaMalloc(&d_compressed_pixels, (num_raw_pixels/8) * sizeof(uint8_t));
    parprog_cudaMemcpy(d_raw_pixels, &raw_pixels, num_raw_pixels * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
    compress_pixels_kernel << <gridDim, blockDim >> > (num_raw_pixels, d_raw_pixels, d_compressed_pixels);
    parprog_cudaMemcpy(&compressed_pixels, d_compressed_pixels, (num_raw_pixels/8) * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    parprog_cudaFree(d_compressed_pixels);
    parprog_cudaFree(d_raw_pixels);

}

void NaiveCudaSimulation::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    calculate_forces(universe, d_positions, d_weights, d_forces);
    calculate_velocities(universe, d_forces, d_weights, d_velocities);
    calculate_positions(universe, d_velocities, d_positions);

    universe.current_simulation_epoch++;
    if(create_intermediate_plots){
        if(universe.current_simulation_epoch % plot_intermediate_epochs == 0){
            std::vector<std::uint8_t> pixels = get_pixels(plotter.get_plot_width(), plotter.get_plot_height(), plotter.get_plot_bounding_box(), d_positions, universe.num_bodies);
            plotter.add_active_pixels_to_image(pixels);

            // This is a dummy to use compression in plotting, although not beneficial performance-wise
            // ----
            // std::vector<std::uint8_t> compressed_pixels;
            // compressed_pixels.resize(pixels.size()/8);
            // compress_pixels(pixels, compressed_pixels);
            // plotter.add_compressed_pixels_to_image(compressed_pixels);
            // ----

            plotter.write_and_clear();
        }
    }
}

void NaiveCudaSimulation::calculate_forces_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_positions, void* d_weights, void* d_forces){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_forces_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_positions, (double*) d_weights, (double2*) d_forces);
}

void NaiveCudaSimulation::calculate_velocities_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_forces, void* d_weights, void* d_velocities){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_velocities_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_forces, (double*) d_weights, (double2*) d_velocities);
}

void NaiveCudaSimulation::calculate_positions_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_velocities, void* d_positions){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_positions_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_velocities, (double2*) d_positions);
}
