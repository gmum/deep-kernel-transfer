""" Code for loading data. """

import numpy as np

INPUT_DIM = 1


class SinusoidalDataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid data.
    A "class" is considered a particular sinusoid function.
    """

    def __init__(self, num_samples_per_class, batch_size, output_dim=1, multidimensional_amp=False,
                 multidimensional_phase=True, noise=True, out_of_range=False):
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
        if out_of_range:
            self.input_range = [-5.0, 10.0]
        self.dim_input = INPUT_DIM
        self.dim_output = output_dim
        self.multidimensional_amp = multidimensional_amp
        self.multidimensional_phase = multidimensional_phase
        self.noise = noise
        #self.split_intervals = [(-5.0, -2.5), (-2.5, 0.0), (0.0, 2.5), (2.5, 5.0)]
        #self.split_intervals = [(-5.0, -3.75), (-3.75, -2.5), (-2.5, -1.25), (-1.25, 0.0),
        #                        (0.0, 1.25), (1.25, 2.5), (2.5, 3.75), (3.75, 5)]

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

        if self.noise == "gaussian" or self.noise =="hetero_multi":
            noise = np.random.normal(0, 0.1, [self.batch_size, self.num_samples_per_class, self.dim_output])
        elif self.noise == "heterogeneous":

            #noise = [np.random.normal(0, 0.1, [self.batch_size, self.num_samples_per_class, self.dim_output]),
            #              np.random.normal(0, 0.2, [self.batch_size, self.num_samples_per_class, self.dim_output]),
            #              np.random.uniform(-0.2, 0.2, [self.batch_size, self.num_samples_per_class, self.dim_output]),
            #              np.random.normal(-0.1, 0.1, [self.batch_size, self.num_samples_per_class, self.dim_output]),]
            noise = [np.random.uniform(-0.1,0.1, [self.batch_size, self.num_samples_per_class, self.dim_output]),
                          np.random.normal(0, 0.75, [self.batch_size, self.num_samples_per_class, self.dim_output]),
                          np.random.uniform(-0.2, 0.2, [self.batch_size, self.num_samples_per_class, self.dim_output]),
                          np.random.normal(0, 0.75, [self.batch_size, self.num_samples_per_class, self.dim_output]),
                          np.random.uniform(-0.2, 0.2, [self.batch_size, self.num_samples_per_class, self.dim_output]),
                          np.random.uniform(-0.1, 0.1, [self.batch_size, self.num_samples_per_class, self.dim_output]),
                          np.random.normal(0, 0.75, [self.batch_size, self.num_samples_per_class, self.dim_output]),
                          np.random.uniform(-0.1, 0.1, [self.batch_size, self.num_samples_per_class, self.dim_output])]
        else:
            noise = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])

        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            numbers = np.random.rand(7) * 10 - 5
            sorted = np.sort(numbers)
            self.split_intervals = [(-5.0, sorted[0])]
            for i in range(1, len(sorted)):
                self.split_intervals += [(sorted[i-1], sorted[i])]
            self.split_intervals += [(sorted[-1], 5.0)]

            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1],
                                                  [self.num_samples_per_class, self.dim_input])
            if input_idx is not None:
                init_inputs[:, input_idx:, 0] = np.linspace(self.input_range[0], self.input_range[1],
                                                            num=self.num_samples_per_class - input_idx, retstep=False)


            outputs[func] = amp[func] * np.sin(init_inputs[func] - phase[func])
            if self.noise == "heterogeneous":
                for i, s in enumerate(self.split_intervals):
                    mask = (init_inputs[func]>=s[0]) & (init_inputs[func]<s[1])
                    outputs[func][mask]=outputs[func][mask]+noise[i][func][mask]
            elif self.noise == "hetero_multi":
                outputs[func] = amp[func] * np.sin(init_inputs[func] + phase[func]) + abs(
                                                            (init_inputs[func] + phase[func])) * noise[func]
            else:
                outputs[func] = outputs[func] + noise[func]
        return init_inputs.astype(np.float32), outputs.astype(np.float32), amp.astype(np.float32), phase.astype(
            np.float32)
