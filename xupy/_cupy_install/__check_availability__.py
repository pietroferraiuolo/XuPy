import os

def _read_code():
    cidir = os.path.abspath(os.path.dirname(__file__))
    icfile = os.path.join(cidir, "__install_cupy__.py")

    with open(icfile) as f:
        code = f.readlines()
    return code

def _was_asked_once():
    code = _read_code()
    return code[4].endswith("True\n")

def xupy_init():
    """
    Subroutine of the XuPy package for the initialization of the GPU support.
    It checks if CuPy is installed and working. If not, it prompts the user to install it.
    It will not prompt the user again if already asked once (it saves the state in __install_cupy__.py).
    
    It is called by xupy._core.py during the import of the package.
    """
    cidir = os.path.abspath(os.path.dirname(__file__))
    icfile = os.path.join(cidir, "__install_cupy__.py")
    if not os.path.exists(icfile):
        print(f"[XuPy] Warning: {icfile} not found. Cannot check for CuPy installation.")
        return

    if _was_asked_once():
        return

    try:
        # Try to import CuPy, to check whether it's installed
        import cupy as xp

        arr = xp.array([1, 2, 3]) # test array
        del arr # cleanup
        import gc
        gc.collect()

    except Exception as err:
        # Prompt user to install CuPy

        print(err,'\n')
        print("[XuPy] GPU Acceleration not available.")
        yn = input("       Attempt to install CuPy? (y/n): ")
        code = _read_code()
        code[4] = f'ASKED_FOR_CUPY = True\n'
        with open(icfile, "w") as f:
            f.writelines(code)
        if yn.lower() not in ['y', 'yes', '']:
            print(f"\n[XuPy] User prompt updated in `{icfile}`.")
            print("       It will not be asked again.")
            print(f"       Install CuPy manually or override the `ASKED_FOR_CUPY` flag in `{icfile}`.")
            return
        else:
            try:
                from xupy._cupy_install.__install_cupy__ import main
                main()
                # Try again after installation

                import cupy as _xp
                import gc

                a = _xp.array([1, 2, 3]) # test array
                del a # cleanup
                gc.collect()
                return
            except Exception as err2:
                print(err2)
                print("\n[XuPy] Failed to install CuPy.")
                return
