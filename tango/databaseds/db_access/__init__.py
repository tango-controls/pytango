__all__list__ = []
def _init_module() :
    import os
    for root,dirs,files in os.walk(__path__[0]) :
        for file_name in files :
            if file_name.startswith('__') : continue
            base,ext = os.path.splitext(file_name)
            if ext == '.py' :
                subdir = root[len(__path__[0]) + 1:]
                if subdir:
                    base = '%s.%s' % (subdir,base)
                __all__list__.append(base)
_init_module()
__all__ = tuple(__all__list__)
