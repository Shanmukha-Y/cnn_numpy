import numpy as np

class Convolution:
    def __init__(self, input_shape, kernel_size, num_filters):
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.kernels = np.random.randn(num_filters, kernel_size, kernel_size, self.input_depth) * 0.1
        self.biases = np.zeros((num_filters, 1))
    
    def forward(self, input_data, stride=1, padding=0):
        self.input = input_data
        self.stride = stride
        self.padding = padding
        
        # Add padding to the input if required
        if padding > 0:
            self.input = np.pad(self.input, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        
        h, w = self.input.shape[:2]
        
        # Calculate output dimensions
        self.out_h = (h - self.kernel_size) // stride + 1
        self.out_w = (w - self.kernel_size) // stride + 1
        
        # Perform convolution
        output = np.zeros((self.out_h, self.out_w, self.num_filters))
        for f in range(self.num_filters):
            for i in range(0, h - self.kernel_size + 1, stride):
                for j in range(0, w - self.kernel_size + 1, stride):
                    output[i//stride, j//stride, f] = np.sum(
                        self.input[i:i+self.kernel_size, j:j+self.kernel_size] * self.kernels[f]
                    ) + self.biases[f]
        
        return output
    
    def backward(self, output_gradient, learning_rate):
        # Initialize gradients
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)
        
        for f in range(self.num_filters):
            for i in range(0, self.out_h):
                for j in range(0, self.out_w):
                    kernels_gradient[f] += self.input[i*self.stride:i*self.stride+self.kernel_size, 
                                                      j*self.stride:j*self.stride+self.kernel_size] * output_gradient[i, j, f]
                    input_gradient[i*self.stride:i*self.stride+self.kernel_size, 
                                   j*self.stride:j*self.stride+self.kernel_size] += self.kernels[f] * output_gradient[i, j, f]
        
        # Update parameters
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=(0, 1)).reshape(self.num_filters, 1)
        
        return input_gradient