#!/usr/bin/python

from math import exp
from random import random
from absl import app, flags
from time import time
from tqdm import trange

FLAGS = flags.FLAGS

def sigmoid(weighted_input):
	return 1.0 / (1.0 + exp(-weighted_input))


DERIVATIVES = {
	sigmoid: lambda output: output * (1.0 - output),
}


class NeuralNetwork(object):
	def __init__(self, structure):
		self._layers = []

		for idx, neurons_count in enumerate(structure):
			if idx == 0:
				layer = [StateNeuron() for i in range(neurons_count)]
			elif idx == len(structure) - 1:
				layer = [PredictionNeuron(sigmoid, FLAGS.learning_rate) for i in range(neurons_count)]
			else:
				layer = [HiddenNeuron(i, sigmoid, FLAGS.learning_rate) for i in range(neurons_count)]

			self._layers.append(layer)

		for idx, layer in enumerate(self._layers):
			for neuron in layer:
				if idx == 0:
					neuron.connect([], self._layers[idx + 1])
				elif idx == len(self._layers) - 1:
					neuron.connect(self._layers[idx - 1], [])
				else:
					neuron.connect(self._layers[idx - 1], self._layers[idx + 1])

	def learn(self, state, solution):
		for idx, value in enumerate(state):
			self._layers[0][idx].output = value

		for idx, value in enumerate(solution):
			self._layers[-1][idx].expected = solution[idx]

		#
		# State forward propagation.
		#
		for idx, neuron in enumerate(self._layers[-1]):
			neuron.calculate_output()

		#
		# Error backward propagation.
		#
		for layer in reversed(self._layers[1:]):
			for neuron in layer:
				neuron.calculate_error()

		#
		# Weights update.
		#
		for layer in self._layers[1:]:
			for neuron in layer:
				neuron.update_weights()

		return [neuron.output for neuron in self._layers[-1]]

	def predict(self, state):
		for idx, value in enumerate(state):
			self._layers[0][idx].output = value

		return [neuron.calculate_output() for neuron in self._layers[-1]]


class Neuron(object):
	def __init__(self):
		self._output = 0.0

		self._output_neurons = None
		self._input_neurons = None

	def connect(self, input_neurons, output_neurons):
		self._input_neurons = input_neurons
		self._output_neurons = output_neurons

	@property
	def output(self):
		return self._output


class StateNeuron(Neuron):
	def __init__(self):
		super(StateNeuron, self).__init__()

	def calculate_output(self):
		return self._output

	@property
	def output(self):
		return super(StateNeuron, self).output

	@output.setter
	def output(self, new_output):
		self._output = new_output


class NeuronCore(Neuron):
	def __init__(self, activation_function, learning_rate):
		super(NeuronCore, self).__init__()

		self._weights = []
		self._error = 0.0

		self._activation_function = activation_function
		self._learning_rate = learning_rate

	@property
	def error(self):
		return self._error

	@property
	def weights(self):
		return self._weights

	def connect(self, input_neurons, output_neurons):
		super(NeuronCore, self).connect(input_neurons, output_neurons)

		for i in range(len(input_neurons)):
			self._weights.append(random())

	def calculate_output(self):
		weighted_input = 0.0

		for idx, neuron in enumerate(self._input_neurons):
			weighted_input += neuron.calculate_output() * self._weights[idx]

		self._output = self._activation_function(weighted_input)

		return self._output

	def update_weights(self):
		new_weights = []
		derivative = DERIVATIVES[self._activation_function](self._output)

		for idx, weight in enumerate(self._weights):
			weight_delta = self._learning_rate * self._input_neurons[idx].output * self._error * derivative

			new_weights.append(self._weights[idx] + weight_delta)

		self._weights = new_weights


class PredictionNeuron(NeuronCore):
	def __init__(self, activation_function, learning_rate):
		super(PredictionNeuron, self).__init__(
			activation_function,
			learning_rate,
		)

		self.expected = 0.0

	def calculate_error(self):
		self._error = self.expected - self._output


class HiddenNeuron(NeuronCore):
	def __init__(self, idx_in_layer, activation_function, learning_rate):
		super(HiddenNeuron, self).__init__(
			activation_function,
			learning_rate,
		)

		self._idx = idx_in_layer

	def calculate_error(self):
		self._error = 0.0

		for neuron in self._output_neurons:
			self._error += neuron.error * neuron.weights[self._idx]


def main(_):
	xor_data_set = (
		((0, 0), [0.0],),
		((0, 1), [1.0],),
		((1, 0), [1.0],),
		((1, 1), [0.0],),
	)

	network_structure = [2, 10, 10, 1]

	network = NeuralNetwork(network_structure)

	start_time = time()

	for i in trange(FLAGS.epochs_count):
		for state, solution in xor_data_set:
			network.learn(state, solution)

	for state, _ in xor_data_set:
		print(state, network.predict(state))

	print("Elapsed time: {}".format(time() - start_time))


if __name__ == "__main__":
	flags.DEFINE_integer("epochs_count", 10000, "Number of epochs.")
	flags.DEFINE_float("learning_rate", 0.9, "Learning rate.")

	app.run(main)
