"""CPU-only convex SAT loops using the already installed Numba runtime.

No fast-math or disk compilation cache. An optional early-stop threshold returns
an upper bound on penetration, so a bound below the limit cannot hide a failure.
"""
import numpy as np
# The installed Numba optional coverage adapter expects a different coverage
# API. This CPU audit does not collect JIT coverage. Suppress that optional
# import only while loading Numba, then restore the Python import table.
# Installed packages and dependency versions remain untouched.
import sys
_missing=object()
_saved_coverage=sys.modules.get('coverage',_missing)
sys.modules['coverage']=None
try:
    from numba import njit
finally:
    if _saved_coverage is _missing:
        sys.modules.pop('coverage',None)
    else:
        sys.modules['coverage']=_saved_coverage


@njit(cache=False)
def _axis_depth(a,b,x,y,z):
    length=(x*x+y*y+z*z)**.5
    if length<=1e-8:return np.inf
    x/=length;y/=length;z/=length
    amin=np.inf;amax=-np.inf;bmin=np.inf;bmax=-np.inf
    for i in range(len(a)):
        value=a[i,0]*x+a[i,1]*y+a[i,2]*z
        amin=min(amin,value);amax=max(amax,value)
    for i in range(len(b)):
        value=b[i,0]*x+b[i,1]*y+b[i,2]*z
        bmin=min(bmin,value);bmax=max(bmax,value)
    return min(amax-bmin,bmax-amin)


@njit(cache=False)
def depths(wa,wb,na,nb,ea,eb,tolerance=0.):
    result=np.zeros(len(wa))
    for t in range(len(wa)):
        best=np.inf
        for normals in (na,nb):
            for j in range(normals.shape[1]):
                axis=normals[t,j]
                best=min(best,_axis_depth(wa[t],wb[t],axis[0],axis[1],axis[2]))
                if best<=tolerance:break
            if best<=tolerance:break
        if best>tolerance:
            for i in range(ea.shape[1]):
                a=ea[t,i]
                for j in range(eb.shape[1]):
                    b=eb[t,j]
                    best=min(best,_axis_depth(wa[t],wb[t],a[1]*b[2]-a[2]*b[1],
                        a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]))
                    if best<=tolerance:break
                if best<=tolerance:break
        result[t]=max(0.,best)
    return result
