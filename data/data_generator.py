
""" Code for loading data. """

import numpy as np


INPUT_DIM=1

class SinusoidalDataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid data.
    A "class" is considered a particular sinusoid function.
    """

    def __init__(self, num_samples_per_class, batch_size, output_dim=1, multidimensional_amp=False,
                 multidimensional_phase=True):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        self.generate = self.generate_sinusoid_batch
        self.amp_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]
        self.input_range = [-5.0, 5.0]
        self.dim_input = INPUT_DIM
        self.dim_output = output_dim
        self.multidimensional_amp = multidimensional_amp
        self.multidimensional_phase = multidimensional_phase

    def generate_sinusoid_batch(self, input_idx=None):
        # input_idx is used during qualitative testing --the number of examples used for the grad update

        if self.multidimensional_amp:
            # y_1 = A_1*sinus(x_1+phi)
            # y_2 = A_2*sinus(x_2+phi)
            # ...
            amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size, self.dim_output])
        else:
            # y_1 = A*sinus(x_1+phi)
            # y_2 = A*sinus(x_2+phi)
            # ...
            amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])

        if self.multidimensional_phase:
            # y_1 = A*sinus(x_1+phi_1)
            # y_2 = A*sinus(x_2+phi_2)
            # ...
            phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size, self.dim_output])
        else:
            # y_1 = A*sinus(x_1+phi)
            # y_2 = A*sinus(x_2+phi)
            # ...
            phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])

        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1],
                                                  [self.num_samples_per_class, self.dim_input])
            if input_idx is not None:
                init_inputs[:, input_idx:, 0] = np.linspace(self.input_range[0], self.input_range[1],
                                                            num=self.num_samples_per_class - input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func] - phase[func])
        return init_inputs.astype(np.float32), outputs.astype(np.float32), amp.astype(np.float32), phase.astype(np.float32)


class PolynomialDataGenerator(object):

    def __init__(self, num_samples_per_class, batch_size, output_dim=1, context=True):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        # CONTEXT - True if want to add the polynomial's degree
        self.context = context

        self.generate = self.generate_polynomial_batch

        # POLYNOMIALS PARAMETERS
        self.polynomial_degrees = [1, 2, 3, 4, 5]
        self.polynomial_coefficients_mu = 0.0
        self.polynomial_coefficients_sigma = 1.0
        self.input_range = [-2.0, 2.0]


        self.dim_input = INPUT_DIM
        self.dim_output = output_dim

    def generate_polynomial_batch(self, input_idx=None):

        degrees = np.random.randint(self.polynomial_degrees[0], self.polynomial_degrees[-1]+1, size=self.batch_size)

        coefficients = np.zeros([self.batch_size, self.polynomial_degrees[-1] + 1, self.dim_input])

        # COEFFICIENTS
        for batch_idx, degree in enumerate(degrees):
            for coefficient_idx in range(degree + 1):
                coefficients[batch_idx, coefficient_idx, 0] = np.random.normal(self.polynomial_coefficients_mu, self.polynomial_coefficients_sigma)

        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])

        one_hot_degrees = np.zeros([self.batch_size, self.polynomial_degrees[-1]])
        for func in range(self.batch_size):
            one_hot_degrees[func, degrees[func] - 1] = 1.0

        # If context is True, we add the the degree of the polynomial
        if self.context:
            # init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input + 1])
            init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input + self.polynomial_degrees[-1]])
        else:
            init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])

        for func in range(self.batch_size):
            if self.context:
                # init_inputs[func] = np.concatenate((np.random.uniform(self.input_range[0], self.input_range[1],
                #                                                   [self.num_samples_per_class, self.dim_input]),
                #                                     np.full([self.num_samples_per_class, self.polynomial_degrees[-1]], degrees[func])), axis=1)
                init_inputs[func] = np.concatenate((np.random.uniform(self.input_range[0], self.input_range[1],
                                                                      [self.num_samples_per_class, self.dim_input]),
                                                    np.full([self.num_samples_per_class, self.polynomial_degrees[-1]], one_hot_degrees[func])), axis=1)
            else:
                init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1],
                                                      [self.num_samples_per_class, self.dim_input])
            if input_idx is not None:
                # if self.context:
                #     init_inputs[:, input_idx:-1, 0] = np.linspace(self.input_range[0], self.input_range[1],
                #                                                   num=self.num_samples_per_class - input_idx, retstep=False)
                # else:
                init_inputs[:, input_idx:, 0] = np.linspace(self.input_range[0], self.input_range[1],
                                                            num=self.num_samples_per_class - input_idx, retstep=False)

        # if self.context:
        #     init_inputs_to_generate = init_inputs[:, : -1, :]
        # else:
        #     init_inputs_to_generate = init_inputs
        init_inputs_to_generate = init_inputs

        for func in range(self.batch_size):
            values = coefficients[func, 0, 0]
            for deg in range(self.polynomial_degrees[0], self.polynomial_degrees[-1] + 1):
                values += coefficients[func, deg, 0] * np.power(init_inputs_to_generate[func, :, 0], deg)
            outputs[func] = values.reshape((self.num_samples_per_class, self.dim_output))

        return init_inputs.astype(np.float32), outputs.astype(np.float32), degrees.astype(np.float32)