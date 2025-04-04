# Tugas Besar 1 IF3270 Pembelajaran Mesin 
# Feedforward Neural Network

# Kelompok 39

# - Dzaky Satrio Nugroho - 13522059
# - Julian Caleb Simandjuntak - 13522099
# - Rafiki Prawhira Harianto - 13522065

# ---------------------------------------------------------------------------------------------------------------

# Import dulu
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------------------------------------------

# Fungsi Aktivasi 

class ActivationFunction:
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.activation_type == 'linear':
            return ActivationFunction.__linear(x)
        elif self.activation_type == 'relu':
            return ActivationFunction.__relu(x)
        elif self.activation_type == 'sigmoid':
            return ActivationFunction.__sigmoid(x)
        elif self.activation_type == 'tanh':
            return ActivationFunction.__tanh(x)
        elif self.activation_type == 'softmax':
            return ActivationFunction.__softmax(x)
        elif self.activation_type == 'leaky_relu':
            return ActivationFunction.__leaky_relu(x)
        elif self.activation_type == 'swish':
            return ActivationFunction.__swish(x)
        else:
            raise ValueError(f"Activation function '{self.activation_type}' not supported")
        
    def backward(self, x: np.ndarray) -> np.ndarray:
        if self.activation_type == 'linear':
            return ActivationFunction.__linear_derivative(x)
        elif self.activation_type == 'relu':
            return ActivationFunction.__relu_derivative(x)
        elif self.activation_type == 'sigmoid':
            return ActivationFunction.__sigmoid_derivative(x)
        elif self.activation_type == 'tanh':
            return ActivationFunction.__tanh_derivative(x)
        elif self.activation_type == 'softmax':
            return ActivationFunction.__softmax_derivative(x)
        elif self.activation_type == 'leaky_relu':
            return ActivationFunction.__leaky_relu_derivative(x)
        elif self.activation_type == 'swish':
            return ActivationFunction.__swish_derivative(x)
        else:
            raise ValueError(f"Activation function '{self.activation_type}' not supported")
    
    def __linear(x: np.ndarray) -> np.ndarray:
        return x

    def __relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def __sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def __tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def __softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def __leaky_relu(x: np.ndarray, alpha=0.1) -> np.ndarray:
        return np.maximum(alpha*x, x)

    def __swish(x: np.ndarray) -> np.ndarray:
        return x * ActivationFunction.__sigmoid(x)

    def __linear_derivative(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    def __relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    
    def __sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        return x * (1 - x)
    
    def __tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - ActivationFunction.__tanh(x) ** 2

    def __softmax_derivative(x: np.ndarray) -> np.ndarray:
        softmax_x = ActivationFunction.__softmax(x)
        return softmax_x * (1 - softmax_x)

    def __leaky_relu_derivative(x: np.ndarray, alpha=0.1) -> np.ndarray:
        return np.where(x > 0, 1, alpha)
    
    def __swish_derivative(x: np.ndarray) -> np.ndarray:
        sigmoidx = ActivationFunction.__sigmoid(x)
        return sigmoidx * (1 + x * (1 - sigmoidx))
    
# ---------------------------------------------------------------------------------------------------------------

# Fungsi Loss

class LossFunction:
    def __init__(self, loss_type: str):
        self.loss_type = loss_type
    
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.loss_type == 'mse':
            return LossFunction.__mse(y_pred, y_true)
        elif self.loss_type == 'bce':
            return LossFunction.__bce(y_pred, y_true)
        elif self.loss_type == 'cce':
            return LossFunction.__cce(y_pred, y_true)
        else:
            raise ValueError(f"Unknown loss function {self.loss_type}")
        
    def loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if self.loss_type == 'mse':
            return LossFunction.__mse_derivative(y_pred, y_true)
        elif self.loss_type == 'bce':
            return LossFunction.__bce_derivative(y_pred, y_true)
        elif self.loss_type == 'cce':
            return LossFunction.__cce_derivative(y_pred, y_true)
        else:
            raise ValueError(f"Unknown loss function {self.loss_type}")
    
    # Mean Squared Error
    def __mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        mse = np.sum((y_true - y_pred) ** 2) / len(y_true)
        return mse

    # Binary Cross-Entropy
    def __bce(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
        return bce

    # Categorical Cross-Entropy
    def __cce(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        cce = -1 / len(y_true) * np.sum(np.sum(y_true * np.log(y_pred)))
        return cce
    
    def __mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return -2 * (y_true - y_pred) / len(y_true) # times dy_pred/dw 
    
    def __bce_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return -1 * (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true)) # times dy_pred/dw 
    
    def __cce_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return -1 * (y_true / (y_pred * len(y_true))) # times dy_pred/dw 

# ---------------------------------------------------------------------------------------------------------------

# Fungsi Inisialisasi Weight

"""
Inisialisasi 1 layer bobot dengan parameter wajib shape yang merupakan tuple berisi ukuran matrix bobot
Contoh: 
shape=(3, 4) berarti:
- Untuk layer dengan 3 neuron awal dan layer dengan 4 neuron berikutnya
- Menghasilkan matrix bobot dengan 4 kolom berdasarkan bias + neuron layer awal dikali 4 kolom berdasarkan neuron layer berikutnya
"""
class WeightInitializer:    
    @staticmethod
    def zeros(shape):
        w = np.zeros(shape)
        b = np.zeros((1, shape[1]))
        return np.vstack((b, w))

    @staticmethod
    def uniform(shape, lower_bound=-0.1, upper_bound=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        w = np.random.uniform(lower_bound, upper_bound, shape)
        b = np.random.uniform(lower_bound, upper_bound, (1, shape[1]))
        return np.vstack((b, w))

    @staticmethod
    def normal(shape, mean=0.0, variance=1.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        std_dev = np.sqrt(variance)  # Konversi variance ke standard deviation
        w = np.random.normal(mean, std_dev, shape)
        b = np.random.normal(mean, std_dev, (1, shape[1]))
        return np.vstack((b, w))
    

    # Kasih penjelasan dikit buat teman2 ku...
    # Apa itu inisialisasi bobot Xavier dan He? 

    # Xavier
    # - Inisialisasi bobot Xavier digunakan untuk menjaga stabilitas varians aktivasi dan varians gradien selama training.
    # - Inisialisasi bobot Xavier melibatkan fan_in (jumlah input / neuron pada input layer) dan fan_out (jumlah output / neuron pada output layer).
    # - Cocok digunakan untuk fungsi aktivasi linear, tanh, atau sigmoid.

    @staticmethod
    def xavier_uniform(shape, seed=None):
        fan_in, fan_out = shape[0], shape[1]
        variance = 2.0 / (fan_in + fan_out)
        limit = np.sqrt(3.0 * variance)
        return WeightInitializer.uniform(shape, lower_bound=-limit, upper_bound=limit, seed=seed)

    @staticmethod
    def xavier_normal(shape, seed=None):
        fan_in, fan_out = shape[0], shape[1]
        variance = 2.0 / (fan_in + fan_out)
        return WeightInitializer.normal(shape, mean=0.0, variance=variance, seed=seed)
    
    # He
    # - Berbeda dengan Xavier, inisialisasi bobot hanya bergantung pada fan_in (jumlah input / neuron pada input layer).
    # - Cocok digunakan untuk fungsi aktivasi ReLU.

    @staticmethod
    def he_uniform(shape, seed=None):
        fan_in = shape[0]
        variance = 2.0 / fan_in
        limit = np.sqrt(3.0 * variance)
        return WeightInitializer.uniform(shape, lower_bound=-limit, upper_bound=limit, seed=seed)

    @staticmethod
    def he_normal(shape, seed=None):
        fan_in = shape[0]
        variance = 2.0 / fan_in
        return WeightInitializer.normal(shape, mean=0.0, variance=variance, seed=seed)
    
    # @staticmethod
    # def initialize_weights(initialization_type: str, shape, bias=1, lower_bound=-0.1, upper_bound=0.1, mean=0.0, variance=1.0, seed=None):
    #     if initialization_type == 'zeros':
    #         return WeightInitializer.zeros(shape, bias=bias)
    #     elif initialization_type == 'uniform':
    #         return WeightInitializer.uniform(shape, bias=bias, lower_bound=lower_bound, upper_bound=upper_bound, seed=seed)
    #     elif initialization_type == 'normal':
    #         return WeightInitializer.normal(shape, bias=bias, mean=mean, variance=variance, seed=seed)
    #     else:
    #         raise ValueError(f"Jenis inisialisasi '{initialization_type}' tidak dikenal.")
    
# Contoh penggunaan
# zero_weights = WeightInitializer.zeros((3,4))
# uniform_weights = WeightInitializer.uniform((3,4))
# normal_weights = WeightInitializer.normal((3,4))
# print(zero_weights)
# print(uniform_weights)
# print(normal_weights)
# output:
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
# [[-0.00770413  0.05834501  0.00577898  0.01360891]
#  [ 0.05610584  0.08511933 -0.08579279 -0.08257414]
#  [-0.07634511 -0.09595632  0.06652397  0.05563135]
#  [ 0.0279842   0.07400243  0.09572367  0.05983171]]
# [[-0.88778575 -2.55298982  0.6536186   0.8644362 ]
#  [-1.98079647 -0.74216502  2.26975462 -1.45436567]
#  [-0.34791215  0.04575852 -0.18718385  1.53277921]
#  [ 0.15634897  1.46935877  0.15494743  0.37816252]]

# ---------------------------------------------------------------------------------------------------------------


# Mencoba membuat FFNN

class FFNN:
    def __init__(self, layers, activations=None, loss="mse", initialization="uniform", seed=0, batch_size=1, learning_rate=0.01, epochs=10, verbose=1, weights=None, regularization=None, lambda_reg=0.01):
        # Parameter-parameter
        # Menerima jumlah neuron dari setiap layer (sekaligus jumlah layernya) termasuk input dan output
        self.layers = layers # Contoh: [1, 2, 3]
        # Menerima fungsi aktivasi tiap layer
        if activations:
            self.activations = [ActivationFunction(activation) for activation in activations]
        else:
            self.activations = [ActivationFunction("sigmoid") for _ in range(len(layers) - 1)]
        # Menerima fungsi loss
        self.loss = LossFunction(loss_type=loss)# Contoh: "mse"
        # Menerima metode inisialisasi bobot
        self.initialization = initialization # Contoh: "zeros"
        self.seed = seed # Jika bobot bukan zeros, menerima seeding
        self.batch_size = batch_size # Jumlah data yang diproses dalam satu iterasi
        self.learning_rate = learning_rate
        self.epochs = epochs # Jumlah iterasi
        self.verbose = verbose # 1 berarti menampilkan progress bar beserta kondisi training loss dan validation loss saat itu, jika 0 tidak usah
        self.value_matrix = []
        self.train_losses = []
        self.val_losses = []
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        
        # Inisialisasi bias dan bobot, beserta gradiennya
        if self.initialization == 'custom':
            self.weights = weights
        else:
            self.weights = []
        self.gradients_w = []

   
        for i in range(1, len(self.layers)):
            in_size, out_size = self.layers[i - 1], self.layers[i]
            shape = (in_size, out_size)

            if self.initialization == 'zeros':
                w = WeightInitializer.zeros(shape)
            elif self.initialization == 'uniform':
                w = WeightInitializer.uniform(shape, seed=self.seed)
            elif self.initialization == 'normal':
                w = WeightInitializer.normal(shape, seed=self.seed)
            elif self.initialization == 'xavier_uniform':
                w = WeightInitializer.xavier_uniform(shape, seed=self.seed)
            elif self.initialization == 'xavier_normal':
                w = WeightInitializer.xavier_normal(shape, seed=self.seed)
            elif self.initialization == 'he_uniform':
                w = WeightInitializer.he_uniform(shape, seed=self.seed)
            elif self.initialization == 'he_normal':
                w = WeightInitializer.he_normal(shape, seed=self.seed)
            elif self.initialization == 'custom':
                continue    
            else:
                raise ValueError("Metode inisialisasi tidak valid.")
            
            self.weights.append(w)

    # Saatnya forward propagation
    def forward_propagation(self, input_data):
        values = np.array(input_data)
        self.value_matrix = [values]

        for i in range(len(self.weights)):
            values = np.insert(values, 0, 1)  # Add bias term
            z = np.dot(self.weights[i].T , values)
            new_values = self.activations[i].forward(z) # Matrix dot multiplication antar weights di layer i dan values
            values = new_values
            self.value_matrix.append(values)
        
        return self.value_matrix[-1] # Print hasil

    def backward_propagation(self, target_output):
        errors = [self.loss.loss_derivative(self.value_matrix[-1], target_output)]
        self.gradients_w = []
        
        for i in reversed(range(len(self.weights))):
            delta = errors[-1] * self.activations[i].backward(self.value_matrix[i + 1])
            prev_values = np.insert(self.value_matrix[i], 0, 1)  # Tambahkan bias
            grad = np.outer(prev_values, delta)
            self.gradients_w.insert(0, grad)
            errors.append(np.dot(self.weights[i], delta)[1:])  # Hilangkan bias dari propagasi ke belakang

    # Menghitung pinalti regularisasi untuk nanti ditambahkan ke loss
    def calculate_regularization_penalty(self):
        penalty = 0
        if self.regularization == "l1":
            for w in self.weights:
                penalty += np.sum(np.abs(w))
        elif self.regularization == "l2":
            for w in self.weights:
                penalty += np.sum(w ** 2)
        return self.lambda_reg * penalty
    
    def update_weights(self):
        for i in range(len(self.weights)):
            gradient = self.gradients_w[i] / self.batch_size
            if self.regularization == "l1":
                gradient += self.lambda_reg * np.sign(self.weights[i])
            elif self.regularization == "l2":
                gradient += 2 * self.lambda_reg * self.weights[i]
            self.weights[i] -= self.learning_rate * gradient

    def train(self, X, y, val_split=0.2):
        X, y = np.array(X), np.array(y)
        
        # Split data
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        if num_samples < 2: # in case train sample is 1
            X_train, y_train = X, y  
            X_val, y_val = np.array([]), np.array([])  
        else:
            split_index = int((1 - val_split) * num_samples)
            X_train, y_train = X[:split_index], y[:split_index]
            X_val, y_val = X[split_index:], y[split_index:]
        
        for epoch in range(self.epochs):
            # if self.verbose:
            #     print(f"Epoch {epoch}")
            total_loss = 0

            num_samples_train = len(X_train)
            indices = np.arange(num_samples_train)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]
            
            for i in range(0, num_samples_train, self.batch_size):
                batch_X = X_train[i:i + self.batch_size]
                batch_y = y_train[i:i + self.batch_size]
                batch_gradients = [np.zeros_like(w) for w in self.weights]
                batch_loss = 0
                
                for j in range(len(batch_X)):
                    self.forward_propagation(batch_X[j])
                    self.backward_propagation(batch_y[j])
                    batch_loss += self.loss.loss(y_pred=self.value_matrix[-1], y_true=batch_y[j])
                    for k in range(len(self.weights)):
                        batch_gradients[k] += self.gradients_w[k]
                
                if self.regularization:
                    batch_loss += self.calculate_regularization_penalty()
                
                self.gradients_w = batch_gradients
                self.update_weights()
                total_loss += batch_loss / len(batch_X)
            
            avg_loss = total_loss / max(num_samples_train / self.batch_size, 1)
            self.train_losses.append(avg_loss)
            
            # Menghitung validation loss
            val_loss = 0
            for i in range(len(X_val)):
                self.forward_propagation(X_val[i])
                val_loss += self.loss.loss(y_pred=self.value_matrix[-1], y_true=y_val[i])
            avg_val_loss = val_loss / max(len(X_val), 1)
            self.val_losses.append(avg_val_loss)

            if self.verbose and epoch % 1 == 0:
                print("Progress: [", end="")
                for i in range (epoch + 1) :
                    print("#", end="")
                for i in range(self.epochs - (epoch + 1)):
                    print("-", end="")
                print(f"] Epoch {epoch + 1}/{self.epochs}")
                    
                print(f"Epoch {epoch + 1}, Loss: {avg_loss:.5f}, Validation Loss: {avg_val_loss:.5f}") 
                
    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label='Training Loss', color='blue')
        plt.plot(range(1, self.epochs + 1), self.val_losses, label='Validation Loss', color='black')
        plt.title('Loss VS Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()          

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            x_pred = self.forward_propagation(x)
            predictions.append(x_pred)
        return predictions
        
    def visualize_weights(self, start=1, end=None, display_size=5, gradient_graph=False):
        if end is None:
            end = len(self.value_matrix)
        max_nodes_in_layer = 0
        
        for i in range(len(self.value_matrix)):
            if len(self.value_matrix[i]) > max_nodes_in_layer:
                max_nodes_in_layer = len(self.value_matrix[i])

        number_of_input = len(self.value_matrix[0])
        number_of_hidden_layer = len(self.value_matrix) - 2
        number_of_output = len(self.value_matrix[len(self.value_matrix)-1])

        nodes = {}

        # Algoritma Nodes
        input_nodes = []
        for i in range(number_of_input):
            input_nodes.append('i' + str(i+1))
        input_nodes.append('b1')

        index_to_layer = {}
        index = 1

        index_to_layer[index] = 'input'
        nodes['input'] = input_nodes
        index += 1

        hidden_nodes = []
        for i in range(number_of_hidden_layer):
            hidden_nodes = []
            for j in range(len(self.weights[i][0])):
                hidden_nodes.append('h' + str(i+1) + '-' + str(j+1))
            hidden_nodes.append('b' + str(i+2))
            nodes['hidden' + str(i+1)] = hidden_nodes
            index_to_layer[index] = 'hidden' + str(i+1)
            index += 1

        output_nodes = []
        for i in range(number_of_output):
            output_nodes.append('o' + str(i+1))

        nodes['output'] = output_nodes
        index_to_layer[index] = 'output'

        edges = []
        edge_labels = {}
        partial_nodes = {}
        max_height = 0

        partial_nodes[index_to_layer[start]] = nodes[index_to_layer[start]]

        # Algoritma label
        for layer_idx in range(start - 1, end - 1):
            if len(nodes[index_to_layer[layer_idx+1]]) > max_height:
                max_height = len(nodes[index_to_layer[layer_idx+1]])

            prev_layer = index_to_layer[layer_idx + 1]
            next_layer = index_to_layer[layer_idx + 2]
            
            partial_nodes[next_layer] = nodes[next_layer]  # Add next layer
            
            weight_matrix = self.gradients_w[layer_idx] if gradient_graph else self.weights[layer_idx]
            
            for prev_idx, prev_neuron in enumerate(nodes[prev_layer]):  
                for next_idx, next_neuron in enumerate(nodes[next_layer]):  
                    if next_neuron.startswith('b'):  # Ignore bias connection
                        continue  
                    
                    edges.append((prev_neuron, next_neuron))  
                    
                    # Bias in first row
                    weight_value = weight_matrix[0][next_idx] if prev_neuron.startswith('b') else weight_matrix[prev_idx + 1][next_idx]  
                    edge_labels[(prev_neuron, next_neuron)] = str(weight_value)[:display_size]  

        print(max_height)
        # Graph
        G = nx.DiGraph()
        G.add_nodes_from(sum(partial_nodes.values(), []))
        G.add_edges_from(edges)

        # Position nodes
        pos = {}

        # Horizontal offset untuk tiap kolom
        x_offset = {}
        for i in range(len(self.value_matrix)):
            x_offset[i] = i+1

        y_positions = {}

        # Algoritma penentuan lokasi nodes
        iterator = 0
        for i in range(start-1, end):
            if i == len(self.value_matrix) - 1:
                y_positions[iterator] = [k for k in range(-1, -len(self.value_matrix[i]) - 1, -1)]
            else:
                y_positions[iterator] = [k for k in range(-1, -len(self.value_matrix[i]) - 2, -1)]
            iterator += 1

        for i, (_, nodes_list) in enumerate(partial_nodes.items()):
            for j, node in enumerate(nodes_list):
                pos[node] = (x_offset[i], y_positions[i][j])

        # Draw graph
        node_color = []
        for node in G.nodes():
            if node.startswith('b'): # Bias
                node_color.append('lightblue')
            elif node.startswith('i'): # Input layer
                node_color.append('lightgreen') 
            elif node.startswith('h'): # Hidden layer
                node_color.append('skyblue')
            elif node.startswith('o'): # Output layer
                node_color.append('orange')

        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=800, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', width=1.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

        # Draw edge labels
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color='red',
            node_size=5500,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
            label_pos=0  # Ensure labels are closer to the source node
        )

        # Menambah title pada gambar
        plt.title("Visualisasi Weight")
        # Set lebar gambar
        plt.xlim(0, end - start + 2)
        # Set tinggi gambar
        plt.ylim(max_height * (-1) - 2, 0)
        # Menghilangkan garis axis
        plt.axis('off')
        # Show the plot
        plt.show()

    def visualize_gradient_weights(self, start=1, end=None, display_size=5):
        self.visualize_weights(start=start, end=end, display_size=display_size, gradient_graph=True)