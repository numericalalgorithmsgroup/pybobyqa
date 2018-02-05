"""
Py-BOBYQA
====

Py-BOBYQA is a general-purpose minimizer which only requires function values.
It is a Python implementation of BOBYQA (Powell, 2009).

It solves the problem:
    min_{x}  f(x),
subject to the (optional) bounds
    lb <= x <= ub,
where f(x) is differentiable and possibly nonconvex.
Since the derivatives of f(x) are never required or approximated,
the solver works when the evaluation of f(x) is noisy.

----

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

The development of this software was sponsored by NAG Ltd. (http://www.nag.co.uk)
and the EPSRC Centre For Doctoral Training in Industrially Focused Mathematical
Modelling (EP/L015803/1) at the University of Oxford. Please contact NAG for
alternative licensing.


"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

from .version import __version__
__all__ = ['__version__']

# Main solver & exit flags
from .solver import *
__all__ += ['solve']

