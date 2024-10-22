from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative on this variable.

        Parameters
        ----------
        x : Any
            Value to accumulate.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable.

        Returns
        -------
        int
            The unique identifier for this variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Check if this variable is a leaf node.

        Returns
        -------
        bool
            True if this is a leaf variable.

        """
        ...

    def is_constant(self) -> bool:
        """Check if this variable is constant.

        Returns
        -------
        bool
            True if the variable is constant.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parent variables of this variable.

        Returns
        -------
        Iterable[Variable]
            Parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to propagate the derivative backward.

        Parameters
        ----------
        d_output : Any
            The upstream derivative.

        Returns
        -------
        Iterable[Tuple[Variable, Any]]
            The chain of variables and their derivatives.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    sorted_variables: List[Variable] = []

    def dfs(v: Variable) -> None:
        """Performs depth-first search for topological sort.

        Parameters
        ----------
        v : Variable
            The current variable in the graph.

        """
        if v in visited or v.is_constant():
            return
        visited.add(v)
        # Recurse on parents
        for parent in v.parents:
            dfs(parent)
        # Append the current variable
        sorted_variables.append(v)

    # Start DFS from the right-most variable
    dfs(variable)

    return reversed(sorted_variables)  # Reverse for topological order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Parameters
    ----------
    variable : Variable
        The variable to backpropagate from.
    deriv : Any
        The derivative to propagate.

    """
    # TODO: Implement for Task 1.4.
    sorted_variables = topological_sort(variable)

    derivatives = {variable: deriv}

    for var in sorted_variables:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var])
        else:
            d_output = derivatives[var]

            for parent, local_derivative in var.chain_rule(d_output):
                if parent in derivatives:
                    derivatives[parent] += local_derivative
                else:
                    derivatives[parent] = local_derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve the saved tensors from the forward pass.

        Returns
        -------
        Tuple[Any, ...]
            The saved values from the forward pass used for backpropagation.

        """
        return self.saved_values
