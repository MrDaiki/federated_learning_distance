import numpy as np

def image_norm(img_a,img_b):

    return np.sqrt(np.sum((img_a-img_b)**2))

"""
alpha need to be positive
"""
def gaussian_kernel(img_a,img_b,alpha):

    return np.exp(-image_norm(img_a,img_b)/(2*alpha**2))

def laplacian_kernel(img_a,img_b,alpha):

    return np.exp(-alpha * ImportWarning(img_a,img_b))


def distance_mmd(distrib_a,distrib_b,alpha=1,sample_size=100):

    mmd_a = 0
    mmd_b = 0

    mmd_ab = 0

    idx_a = np.random.choice(np.arange(len(distrib_a)),size=sample_size,replace=True)

    idx_b = np.random.choice(np.arange(len(distrib_b)),size=sample_size,replace=True)

    a_sample = distrib_a[idx_a]
    b_sample = distrib_b[idx_b]

    for i in range(sample_size):

        for j in range(sample_size):
            
            mmd_ab += gaussian_kernel(a_sample[i],b_sample[j],alpha)

            if i!=j:

                mmd_a += gaussian_kernel(a_sample[i],a_sample[j],alpha)
                mmd_b += gaussian_kernel(b_sample[i],b_sample[j],alpha)

    return (mmd_a + mmd_b)/(sample_size*(sample_size-1)) - 2*mmd_ab/sample_size**2


