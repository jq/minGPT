import torch
import numpy as np

def tensor_demo():
    print(torch.__version__)
    # A scalar is a single number and in tensor-speak it's a zero dimension tensor.
    scalar = torch.tensor(7)
    print(scalar.item())
    vector = torch.tensor([7, 7])
    print(vector.shape)
    MATRIX = torch.tensor([[7, 8],
                           [9, 10]])
    print(MATRIX.shape)
    TENSOR = torch.tensor([[[1, 2, 3],
                            [3, 6, 9],
                            [2, 4, 5]]])
    # 1 dimension of 3 by 3.
    print(TENSOR.shape)

    random_tensor = torch.rand(size=(3, 4))
    print(random_tensor, random_tensor.dtype)
    zeros = torch.zeros(size=(3, 4))
    ones = torch.ones(size=(3, 4))
    zero_to_ten = torch.arange(start=0, end=10, step=1)
    print(zero_to_ten)
    # 10 zeros with the same shape as zero_to_ten
    ten_zeros = torch.zeros_like(input=zero_to_ten)
    float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
       dtype=torch.float16, # defaults to None, which is torch.float32 or whatever datatype is passed
       device=None, # GPU or CPU
       requires_grad=False) # if True, operations performed on the tensor are recorded
    ten_ten = ten_zeros + 10
    # Element-wise matrix multiplication
    print(ten_ten * ten_ten)

    # Matrix multiplication
    print(torch.matmul(ten_ten, ten_ten))
    # Shapes need to be in the right way
    tensor_A = torch.tensor([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=torch.float32)

    tensor_B = torch.tensor([[7, 10],
                             [8, 11],
                             [9, 12]], dtype=torch.float32)
    torch.mm(tensor_A, tensor_B.T)

    # Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
    torch.manual_seed(42)
    # This uses matrix multiplication
    linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input
                             out_features=6) # out_features = describes outer value
    x = tensor_A
    output = linear(x)
    print(f"Input shape: {x.shape}\n")
    print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
    x = torch.arange(0, 100, 10)
    print(f"Minimum: {x.min()}")
    print(f"Maximum: {x.max()}")
    # print(f"Mean: {x.mean()}") # this will error
    print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
    print(f"Sum: {x.sum()}")
    tensor_float16 = x.type(torch.float16)

    # reshape the tensor, totoal number of elements must be the same
    ten_zeros_reshape = ten_zeros.reshape(2, 5)
    ten_zeros[0] = 5
    print(ten_zeros)
    # https://stackoverflow.com/a/54507446/7900723
    # view vs reshape: view the new tensor will always share its data with the original tensor
    # so it imposes some contiguity constraints on the shapes of the two tensors,
    # reshape doesn't impose any contiguity constraints, but also doesn't guarantee data sharing.
    # The new tensor may be a view of the original tensor, or it may be a new tensor altogether.
    # dim = 1 stack along the column, dim = 0 stack along the row
    x_stacked = torch.stack([x, x, x, x], dim=1)
    print(x_stacked)

    x_reshaped = x.reshape(1, len(x))
    print(f"Previous tensor: {x_reshaped}")
    print(f"Previous shape: {x_reshaped.shape}")

    # Remove extra dimension from x_reshaped, torch.unsqueeze() adds a dimension of 1
    x_squeezed = x_reshaped.squeeze()
    print(f"\nNew tensor: {x_squeezed}")
    print(f"New shape: {x_squeezed.shape}")

    x_original = torch.rand(size=(224, 224, 3))
    # Permute the original tensor to rearrange the axis order
    x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
    print(f"Previous shape: {x_original.shape}")
    print(f"New shape: {x_permuted.shape}")

def index():
    x = torch.arange(1, 10).reshape(1, 3, 3)
    print(x[0])
    # 取 【x,y】中 Y = 1
    print(x[0][:,1])
    print(x[0][1,:])

# Get all values of 0th dimension and the 0 index of 1st dimension
    # 只有一个[] 所以output是[[]] 是取[[]]的0, == print(x[:, 0,:])
    print(x[:, 0])
    print(x[:, 1])
    # Get all values of 0th & 1st dimensions and index 1 of 2nd dimension
    # this is the col 1
    print(x[:, :, 1])
    # Get all values of the 0 dimension and the 1 index value of the 1st and 2nd dimension
    print(x[:, 1, 1])
def numpy():
    array = np.arange(1.0, 8.0)
    tensor = torch.from_numpy(array)
    numpy_tensor = tensor.numpy()

def rands():
    # # Set the random seed
    RANDOM_SEED=42 # try changing this to different values and see what happens to the numbers below
    torch.manual_seed(seed=RANDOM_SEED)
    random_tensor_C = torch.rand(3, 4)

    # Have to reset the seed every time a new rand() is called
    # Without this, tensor_D would be different to tensor_C
    torch.random.manual_seed(seed=RANDOM_SEED) # try commenting this line out and seeing what happens
    random_tensor_D = torch.rand(3, 4)

    print(f"Tensor C:\n{random_tensor_C}\n")
    print(f"Tensor D:\n{random_tensor_D}\n")
    print(f"Does Tensor C equal Tensor D? (anywhere)")
    random_tensor_C == random_tensor_D


def gpu():
    # Set device type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.device_count()
    tensor = torch.tensor([1, 2, 3])

    # Tensor not on GPU
    print(tensor, tensor.device)

    # Move tensor to GPU (if available)
    tensor_on_gpu = tensor.to(device)
    tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()

def exer1():R

if __name__ == '__main__':
    index()