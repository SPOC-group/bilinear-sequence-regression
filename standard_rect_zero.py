import numpy as np
import sys
import torch
from torch import nn



class NetworkTeacher(nn.Module):
    def __init__(self, D, M, L):
        super(NetworkTeacher, self).__init__()
        self.D = D
        self.M = M        
        self.L = L
        self.fc1U = nn.Parameter(torch.normal(0, 1, (M, D), requires_grad=True))
        self.fc1V = nn.Parameter(torch.normal(0, 1, (L, M), requires_grad=True))


    def forward(self, x):
        S = self.fc1V @ self.fc1U
        return torch.einsum("ij,nij", S, x) / np.sqrt(self.M) / np.sqrt(self.L * self.D) 
       


class NetworkStudent(nn.Module):
    def __init__(self, D, M, L):
        super(NetworkStudent, self).__init__()
        self.D = D
        self.M = M     
        self.L = L   
        self.fc1U = nn.Parameter(torch.normal(0, 1e-4, (M, D), requires_grad=True))
        self.fc1V = nn.Parameter(torch.normal(0, 1e-4, (D, M), requires_grad=True))


    def forward(self, x):
        S = self.fc1V @ self.fc1U
        return torch.einsum("ij,nij", S, x) / np.sqrt(self.M) / np.sqrt(self.L * self.D) 
       


def main(D, alpha, rho, beta, L, lr, T, samples):
    M = int(D * rho)
    M_star = int(D * rho)
    N = int(D*L * alpha)

    gen_error = np.ones((samples, T))
    train_error = np.ones((samples, T))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in range(samples):
        # Initialise the teacher and student networks
        with torch.no_grad():
            teacher = NetworkTeacher(D, M_star, L).to(device)
        
        with torch.no_grad():
            # The network is trained on a random dataset of size N
            X = torch.normal(0,1, (N, L,D), requires_grad=False).to(device)
            y = teacher(X)
        

            student = NetworkStudent(D, M, L).to(device)
            
            # Optimizer
            optimizer = torch.optim.SGD(student.parameters(), lr=lr)
            
        # The training loop
        for t in range(T):

            # Compute the generalization error
            with torch.no_grad():

                # retrieve the data from the GPU
                U = student.fc1U.data.cpu().numpy()
                V = student.fc1V.data.cpu().numpy()
                U_star = teacher.fc1U.data.cpu().numpy()
                V_star = teacher.fc1V.data.cpu().numpy()

                S = V @ U / np.sqrt(M)
                S_star = V_star @ U_star / np.sqrt(M_star)
                
                gen_error[s,t] = np.mean((S - S_star)**2)

                # if t % 100 == 0:
                #     print(f"Sample {s+1}/{samples}, Iteration {t+1}/{T}, Generalization error: {gen_error[s,t]}, Trace difference: {np.trace(S)/D - np.trace(S_star)/D}, Norm S: {np.linalg.norm(S)}")


                
            # Compute the gradient of the loss with respect to the student network parameters
            y_pred = student(X)
            loss = ((y_pred - y)**2).sum()/4

            with torch.no_grad():
                train_error[s,t] = loss.cpu().item()/N*4

            loss.backward()

            if t % 100 == 0:
                print(f"Alpha {alpha}, Sample {s+1}/{samples}, Iteration {t+1}/{T}, Generalization error: {gen_error[s,t]}, Train error: {train_error[s,t]}")

            # Update the student network parameters
            optimizer.step()
            optimizer.zero_grad()



        
    # Save the results
    np.save(f"standard/zero_gen_error_{D}_{alpha}_{rho}_{beta}_{lr}.npy", gen_error)
    np.save(f"standard/zero_train_error_{D}_{alpha}_{rho}_{beta}_{lr}.npy", train_error)



if __name__ == '__main__':
    # Get the parameters from the command line
    D = int(sys.argv[1])
    alpha = float(sys.argv[2]) 
    rho = float(sys.argv[3])
    beta = float(sys.argv[4])


    L = D // beta
    rho = rho / beta
    

    T = int(50000)
    samples = 3
    lr = 0.1

    main(D, alpha, rho, beta, L, lr, T, samples)

