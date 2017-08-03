#! /usr/bin/env python

"""
This is a library of misc. file utility classes

History:

Date:           20051005
Version:        1.0.0.0
Author:         John Jorgensen
Description:    Initial implementation
"""

version = '1.0.0.0'

import os,shutil,fnmatch

def unixpath(path):
    return path.replace('\\','/')
    
def match(filename,patterns):
    for pattern in patterns:
        if fnmatch.fnmatch(filename,pattern):
            return True
    return False

def copyfile(src,dst,excludefiles=[]):
        fp,fn = os.path.split(src)
        if not match(fn,excludefiles):
            shutil.copyfile(src,dst)
                        
def copytree(src,dst,excludefiles=[]):
    filenames = os.walk(src)
    for folder,folders,files in filenames:
        fp,fn = os.path.split(folder)
        if match(fn,excludefiles):
            excludefiles.append(folder)
            continue
        elif match(fp,excludefiles):
            excludefiles.append(folder)
            continue
        else:
            srcpaths = [os.path.join(folder,fn) for fn in files if not match(fn,excludefiles)]
            commonprefix = os.path.commonprefix([folder,src])
            subpath = folder.replace(commonprefix,'')[1:]
            for srcpath in srcpaths:
                sp,sn = os.path.split(srcpath)
                dstpath = os.path.join(dst,subpath,sn)
                dp,dn = os.path.split(dstpath)
                try:
                    os.makedirs(dp)
                except:
                    pass
                shutil.copyfile(srcpath,dstpath)
            else:
                # Create empty folders
                if subpath <> '':
                    dp = os.path.join(dst,subpath)
                    try:
                        os.makedirs(dp)
                    except:
                        pass
    
    
if __name__ == '__main__':

    copytree(r'\\dtc0\radac\daq\shell\v10.1.0.0',r'c:\tmp\copytreetest',excludefiles=['*.dcu','.svn','*.~*'])

