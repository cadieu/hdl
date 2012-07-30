import numpy as np

def whiten_var(P,num_eigs=None,perc_var=None):
    """
    Compute the whitened P using the principal components of P.
        P is a numVars x numObservations numeric array.

    Return value are
    whiteP      : a num components x num observations numeric array
    W           : the whitening matrix, ie, whiteP = W*P
    unW         : the unwhitening matrix, ie, P^ = unW*whiteP
    zeroW       : zero-phase whitening

    """
    Pmean = np.zeros((P.shape[0],1))
    # get eigs and eigenvectors
    for i in range(P.shape[0]):
        Pmean[i] = P[i,:].mean()
        P[i,:] -= Pmean[i]
    d, E = np.linalg.eigh(np.dot(P,P.T)/P.shape[1])

    if (num_eigs is None and perc_var is None) or (perc_var >= 100):
        order = np.argsort(d)[::-1]
    elif perc_var is not None:
        order = np.argsort(d)[::-1]
        d_var = np.cumsum(np.abs(d[order]))/np.sum(np.abs(d[order]))
        num_eigs = np.argmax(d_var>float(perc_var)/100.)
        order = order[:num_eigs]
        print '%d Eigs needed for %f perc. of variance'%(num_eigs,perc_var)
    else:
        order = np.argsort(d)[::-1]
        order = order[:num_eigs]

    E = E[:,order]
    D = np.diag(np.real(d[order]**(-0.5)))
    whiten = np.dot(D,E.T)
    dewhiten = np.dot(np.diag(np.real(d[order]**0.5)),E.T).T

    zerowhiten = np.dot(E,whiten)
    zerodewhiten = np.dot(dewhiten,E.T)
    whiteP = np.dot(whiten,P)

    return whiteP, Pmean, whiten, dewhiten, zerowhiten, zerodewhiten