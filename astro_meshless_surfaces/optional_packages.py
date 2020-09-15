"""
Optional packages handling.
"""

try:
    from numba import jit, prange

    # from numba.typed import List

except (ImportError, ModuleNotFoundError):
    print("No numba found. Consider installing it: pip install numba")

    def jit(*args, **kwargs):
        def x(func):
            return func

        return x

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

    # List = tuple


#
#  def jit(*args, **kwargs):
#      def x(func):
#          return func
#
#      return x
#
#  def prange(*args, **kwargs):
#      return range(*args, **kwargs)
#
#  List = list
