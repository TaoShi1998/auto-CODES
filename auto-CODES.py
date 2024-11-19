import torch

    
def corr(f, g):
    k = torch.mean(torch.sum(f * g, 1)) 
    return k 
    

def cov_trace(f, g):
    cov_f = f.T @ f / (f.shape[0] - 1)
    cov_g = g.T @ g / (g.shape[0] - 1)
    return torch.trace(cov_f @ cov_g)


def neg_hscore(f, g):
    output = -corr(f, g) + cov_trace(f, g) / 2
    return output 


def mcr_loss(f, g, p, alpha, f_extend, g_extend, device):
    num_class = len(p)
    loss_term = torch.zeros(num_class).to(device)
    for i in range(len(f_extend)):
        loss_term[i] = p[i] * torch.sum(torch.mean(f_extend[i], 0) * torch.mean(g_extend[i], 0))
    loss_term = torch.sum(loss_term)
    loss = 2 * (1 - alpha) * neg_hscore(f, g) + alpha * cov_trace(f, g) - 2 * alpha * loss_term
    return loss


def update_alpha(f, g, f_extend, p, n, label, avail_label, dim, alpha_truncate):
    num_class=len(p)
    label = label.cuda()
    
    Gamma = (torch.mm(f.T, f)/n)
    Gi = torch.inverse(Gamma)

    Ax = []
    for k in range(num_class):
            Ax.append(f[label==avail_label[k]])
    
    Ay = []
    for k in range(num_class):
            Ay.append(g[label==avail_label[k]])
        
    Ax_extend = []
    for k in range(num_class):
        Ax_extend.append(f_extend[k])
        
    ny = []
    for k in range(num_class):
        ny.append(f[label==avail_label[k]].size()[0])
        
    Ax1 = []
    for k in range(num_class):
        temp = Ax[k].unsqueeze(1) * torch.ones([1, ny[k], 1]).cuda()
        temp = torch.reshape(temp, [ny[k]*ny[k], dim])
        Ax1.append(temp)
        
    A1x = []
    for k in range(num_class):
        temp = Ax[k].unsqueeze(0) * torch.ones([ny[k], 1, 1]).cuda()
        temp = torch.reshape(temp, [ny[k]*ny[k], dim])
        A1x.append(temp)
        
    term1 = 0
    for k in range(num_class):
        temp = torch.trace(torch.mm(Gi, torch.mm(Ax[k].T, Ax[k])))/ny[k]
        term1 = term1 + (1/n - 1/(p[k]*n**2))*temp
    
    term2 = 0
    for k in range(num_class):
        temp = - torch.mm(torch.mm(torch.mean(Ax[k],0, keepdim=True), Gi) , torch.mean(Ax[k],0, keepdim=True).reshape((dim,1)))
        term2 = term2 + (p[k]/n - 1/(p[k]*n**2))*temp
    
    term3 = 0
    for k in range(num_class):
        temp = torch.mm(torch.mm(torch.mean(Ax_extend[k],0, keepdim=True), Gi) , torch.mean(Ax[k],0, keepdim=True).reshape((dim,1)))
        term3 = term3 + (p[k]/n + 1/n - 2/(p[k]*n**2))*temp
        
    term4 = 0
    for k in range(num_class):
        temp = - torch.trace(torch.mm(Gi, torch.mm(Ax1[k].T, Ax_extend[k])))/(ny[k]**2)
        term4 = term4 + (1/n - 1/(p[k]*n**2))*temp
    
    term5 = 0
    for k in range(num_class):
        temp = - torch.trace(torch.mm(Gi, torch.mm(A1x[k].T, Ax_extend[k])))/(ny[k]**2)
        term5 = term5 + (1/n - 1/(p[k]*n**2))*temp    

    up = term1 + term2 + term3 + term4 + term5
    
    term6 = 0
    for k in range(num_class):
        temp =  torch.trace(torch.mm(Gi, torch.mm(Ax[k].T, Ax[k])))/ny[k]
        term6 = term6 + temp/n
        
    term7 = 0
    for k in range(num_class):
        temp = - torch.mm(torch.mm(torch.mean(Ax_extend[k],0, keepdim=True), Gi) , torch.mean(Ax[k],0, keepdim=True).reshape((dim,1)))
        term7 = term7 + (2*(n-1)*p[k]/n - 6/n)*temp
        
    term8 = 0
    for k in range(num_class):
        temp = torch.mm(torch.mm(torch.mean(Ax[k],0, keepdim=True), Gi) , torch.mean(Ax[k],0, keepdim=True).reshape((dim,1)))
        term8 = term8 + ((n-1)*p[k]/n - 2/n)*temp
        
    term9 = 0
    for k in range(num_class):
        temp = torch.trace(torch.mm(Gi, torch.mm(Ax1[k].T, Ax_extend[k])))/(ny[k]**2)
        term9 = term9  - (2/n)*temp
        
    term10 = 0
    for k in range(num_class):
        temp = torch.trace(torch.mm(Gi, torch.mm(A1x[k].T, Ax_extend[k])))/(ny[k]**2)
        term10 = term10  - (2/n)*temp
        
    term11 = 0
    for k in range(num_class):
        temp =  torch.mm(torch.mm(torch.mean(Ax_extend[k],0, keepdim=True), Gi) , torch.mean(Ax_extend[k],0, keepdim=True).reshape((dim,1)))
        term11 = term11 + ((n-1)*p[k]/n - 5/n)*temp
    
    term12 = 0
    for k in range(num_class):
        Ax_extend_reshape = Ax_extend[k].reshape([ny[k], ny[k], dim])
        temp = []
        for k3 in range(ny[k]):
          Ax_extend_k3 = (Ax_extend_reshape[:,k3:k3+1,:] + torch.zeros([ny[k], ny[k], dim]).cuda()).reshape([ny[k]**2, dim])
          temp.append(torch.mm(Ax_extend[k].T, Ax_extend_k3)/(ny[k]**2))
        temp = torch.stack(temp, 0)
        temp = torch.mean(temp, 0)
        temp = torch.trace(torch.mm(Gi, temp))
        term12 = term12 + (1/n)*temp
        
    term13 = 0
    for k in range(num_class):
        Ax_extend_reshape = Ax_extend[k].reshape([ny[k], ny[k], dim])
        temp = []
        for k3 in range(ny[k]):
          Ax_extend_k3 = (Ax_extend_reshape[k3:k3+1,:,:] + torch.zeros([ny[k], ny[k], dim]).cuda()).reshape([ny[k]**2, dim])
          temp.append(torch.mm(Ax_extend[k].T, Ax_extend_k3)/(ny[k]**2))
        temp = torch.stack(temp, 0)
        temp = torch.mean(temp, 0)
        temp = torch.trace(torch.mm(Gi, temp))
        term13 = term13 + (1/n)*temp

    term14 = 0
    for k in range(num_class):
        Ax_extend_reshape = Ax_extend[k].reshape([ny[k], ny[k], dim])
        temp = []
        for k3 in range(ny[k]):
          Ax_extend_k3 = (Ax_extend_reshape[k3:k3+1,:,:].permute(1,0,2) + torch.zeros([ny[k], ny[k], dim]).cuda()).reshape([ny[k]**2, dim])
          temp.append(torch.mm(Ax_extend[k].T, Ax_extend_k3)/(ny[k]**2))
        temp = torch.stack(temp, 0)
        temp = torch.mean(temp, 0)
        temp = torch.trace(torch.mm(Gi, temp))
        term14 = term14 + (1/n)*temp
        
    down = term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13 + term14*2
    
    alpha = up/down
    
    if alpha_truncate:
        if alpha < 0:
            alpha = 0
        if alpha > 1:
            alpha = 1

    return alpha, up, down
