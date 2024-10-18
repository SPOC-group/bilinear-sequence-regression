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
        # self.fc1U = nn.Parameter(torch.normal(0, 1e-4, (M, D), requires_grad=True))
        # self.fc1V = nn.Parameter(torch.normal(0, 1e-4, (D, M), requires_grad=True))
        self.fc1U = nn.Parameter(torch.normal(0, 1, (M, D), requires_grad=True))
        self.fc1V = nn.Parameter(torch.normal(0, 1, (L, M), requires_grad=True))


    def forward(self, x):
        S = self.fc1V @ self.fc1U
        return torch.einsum("ij,nij", S, x) / np.sqrt(self.M) / np.sqrt(self.L * self.D) 


def main(D, alpha, beta, beta_star, L, lr, T, samples, averages):
    M = int(D * beta)
    M_star = int(D * beta_star)
    N = int(D*L * alpha)

    gen_error = np.ones((samples))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S_all = np.zeros((averages, L, D))
    for s in range(samples):
        # Initialise the teacher and student networks
        with torch.no_grad():
            teacher = NetworkTeacher(D, M_star, L).to(device)

            U_star = teacher.fc1U.data.cpu().numpy()
            V_star = teacher.fc1V.data.cpu().numpy()
            S_star = V_star @ U_star / np.sqrt(M_star)

            X = torch.normal(0,1, (N, L,D), requires_grad=False).to(device)
            y = teacher(X)
        
        
        for av in range(averages):
            student = NetworkStudent(D, M, L).to(device)
            
            # Optimizer
            optimizer = torch.optim.SGD(student.parameters(), lr=lr)
            
            # The training loop
            for t in range(T):               
                # Compute the gradient of the loss with respect to the student network parameters
                y_pred = student(X)
                loss = ((y_pred - y)**2).sum()/4

                loss.backward()


                # Update the student network parameters
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                U = student.fc1U.data.cpu().numpy()
                V = student.fc1V.data.cpu().numpy()

                S = V @ U / np.sqrt(M)
                

            S_all[av] = S
            S_averaged = S_all[:av+1].mean(axis=0)
            print(f"Sample {s+1}/{samples}, Iteration {av+1}/{averages}, Generalization error: {np.mean((S_averaged - S_star)**2)}")

        gen_error[s] = np.mean((S_averaged - S_star)**2)

        
    # Save the results
    np.save(f"data_averaged/gen_error_{D}_{alpha}_{beta}_{beta_star}_{lr}.npy", gen_error)



if __name__ == '__main__':
    # Get the parameters from the command line
    D = int(sys.argv[1])
    alpha = float(sys.argv[2]) 
    beta = float(sys.argv[3])
    beta_star = float(sys.argv[4])


    L = D // 2
    # L = D
    
    T = int(50000)
    samples = 4
    lr = 0.1

    main(D, alpha, beta, beta_star, L, lr, T, samples, averages=64)
