"""
Scope and variable analysis for refactor-snipe.

Ported from refactoring.nvim's ts-locals.lua and treesitter.lua.
This module handles:
- Finding containing scopes (functions, classes, modules)
- Tracking variable definitions and references
- Identifying what variables are used in a region
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tree_sitter import Node

try:
    from .treesitter import Region, TreeSitterParser
except ImportError:
    from treesitter import Region, TreeSitterParser


@dataclass
class VariableInfo:
    """Information about a variable definition or reference."""

    name: str
    node: Node
    definition_node: Node | None = None  # For references, points to the definition


@dataclass
class ScopeAnalyzer:
    """Analyzes scopes and variables in parsed code."""

    parser: TreeSitterParser
    # Node types that define scopes in Python
    scope_types: set[str] = field(
        default_factory=lambda: {
            "module",
            "function_definition",
            "class_definition",
            "lambda",
        }
    )
    # Node types that define blocks (for extract block)
    block_types: set[str] = field(
        default_factory=lambda: {
            "block",
            "function_definition",
            "module",
        }
    )

    def get_containing_scope(self, node: Node) -> Node | None:
        """Find the innermost scope containing a node."""
        current = node.parent
        while current is not None:
            if current.type in self.scope_types:
                return current
            current = current.parent
        return None

    def get_scope_chain(self, node: Node) -> list[Node]:
        """Get all scopes containing a node, from innermost to outermost."""
        scopes = []
        current = node
        while current is not None:
            if current.type in self.scope_types:
                scopes.append(current)
            current = current.parent
        return scopes

    def get_function_scope(self, node: Node) -> Node | None:
        """Find the containing function definition."""
        # If node itself is a function, return it
        if node.type == "function_definition":
            return node
        current = node.parent
        while current is not None:
            if current.type == "function_definition":
                return current
            current = current.parent
        return None

    def get_class_scope(self, node: Node) -> Node | None:
        """Find the containing class definition."""
        current = node.parent
        while current is not None:
            if current.type == "class_definition":
                return current
            current = current.parent
        return None

    def is_inside_class(self, node: Node) -> bool:
        """Check if a node is inside a class definition."""
        return self.get_class_scope(node) is not None

    def get_function_parameters(self, func_node: Node) -> list[str]:
        """Get parameter names from a function definition."""
        if func_node.type != "function_definition":
            return []

        params = []
        # Query for parameters
        query = """
        (function_definition
          parameters: (parameters
            [(identifier) @param
             (default_parameter name: (identifier) @param)
             (typed_parameter (identifier) @param)
             (typed_default_parameter name: (identifier) @param)]))
        """

        for node, name in self.parser.query_captures(query, func_node):
            if name == "param":
                param_text = self.parser.get_text(node)
                if param_text != "self":
                    params.append(param_text)

        return params

    def get_local_assignments(self, scope: Node) -> list[tuple[str, Node]]:
        """Get all local variable assignments in a scope."""
        assignments = []

        # Query for assignments
        query = """
        (assignment
          left: [(identifier) @name
                 (pattern_list (identifier) @name)
                 (tuple_pattern (identifier) @name)])
        """

        for node, name in self.parser.query_captures(query, scope):
            if name == "name":
                var_name = self.parser.get_text(node)
                if var_name != "self" and not var_name.startswith("self."):
                    assignments.append((var_name, node))

        return assignments

    def get_references_in_region(self, region: Region, scope: Node) -> list[tuple[str, Node]]:
        """Get all variable references within a region."""
        references = []

        # Query for identifiers that are references (not definitions)
        query = "(identifier) @ref"

        for node, _ in self.parser.query_captures(query, scope):
            if region.contains_node(node):
                # Skip if this is part of a definition (left side of assignment)
                parent = node.parent
                if parent and parent.type == "assignment":
                    # Check if this is on the left side
                    left = parent.child_by_field_name("left")
                    if left and self._node_contains(left, node):
                        continue

                # Skip if it's a function/class name definition
                if parent and parent.type in ("function_definition", "class_definition"):
                    name_node = parent.child_by_field_name("name")
                    if name_node and name_node.id == node.id:
                        continue

                # Skip if it's a parameter definition
                if parent and parent.type in (
                    "parameters",
                    "default_parameter",
                    "typed_parameter",
                    "typed_default_parameter",
                ):
                    continue

                references.append((self.parser.get_text(node), node))

        return references

    def _node_contains(self, container: Node, target: Node) -> bool:
        """Check if container node contains target node."""

        def walk(node: Node) -> bool:
            if node.id == target.id:
                return True
            for child in node.children:
                if walk(child):
                    return True
            return False

        return walk(container)

    def get_references_after_region(self, region: Region, scope: Node) -> list[tuple[str, Node]]:
        """Get all variable references that come after a region within a scope."""
        references = []
        query = "(identifier) @ref"

        for node, _ in self.parser.query_captures(query, scope):
            if region.is_after(node):
                continue  # Skip nodes before/in the region

            node_region = Region.from_node(node)
            if node_region.start.row > region.end.row or (
                node_region.start.row == region.end.row and node_region.start.col >= region.end.col
            ):
                references.append((self.parser.get_text(node), node))

        return references

    def find_variables_defined_in_region(self, region: Region, scope: Node) -> set[str]:
        """Find all variables that are defined within a region."""
        defined = set()

        query = """
        (assignment
          left: [(identifier) @name
                 (pattern_list (identifier) @name)
                 (tuple_pattern (identifier) @name)])
        """

        for node, name in self.parser.query_captures(query, scope):
            if name == "name" and region.contains_node(node):
                var_name = self.parser.get_text(node)
                if var_name != "self":
                    defined.add(var_name)

        # Also check for loop variables
        query_for = """
        (for_statement
          left: [(identifier) @name
                 (pattern_list (identifier) @name)
                 (tuple_pattern (identifier) @name)])
        """

        for node, name in self.parser.query_captures(query_for, scope):
            if name == "name" and region.contains_node(node):
                var_name = self.parser.get_text(node)
                if var_name != "self":
                    defined.add(var_name)

        return defined

    def find_variables_used_in_region(self, region: Region, scope: Node) -> set[str]:
        """Find all variables that are used (referenced) within a region."""
        refs = self.get_references_in_region(region, scope)
        return {name for name, _ in refs}

    def find_free_variables_in_region(self, region: Region, scope: Node) -> set[str]:
        """
        Find variables used in a region that are defined outside the region.
        These are the variables that need to be passed as parameters when extracting.
        """
        used = self.find_variables_used_in_region(region, scope)
        defined_in_region = self.find_variables_defined_in_region(region, scope)

        # Also get function parameters if we're in a function
        func_scope = self.get_function_scope(scope)
        if func_scope:
            params = set(self.get_function_parameters(func_scope))
        else:
            params = set()

        # Get assignments before the region in the same scope
        assignments_before = set()
        for var_name, node in self.get_local_assignments(scope):
            node_region = Region.from_node(node)
            if node_region.end.row < region.start.row or (
                node_region.end.row == region.start.row and node_region.end.col < region.start.col
            ):
                assignments_before.add(var_name)

        # Free variables = used - defined_in_region, filtered to known definitions
        free = used - defined_in_region
        known = params | assignments_before

        return free & known

    def find_return_variables(self, region: Region, scope: Node) -> list[str]:
        """
        Find variables defined in the region that are used after the region.
        These need to be returned from the extracted function.
        """
        defined_in_region = self.find_variables_defined_in_region(region, scope)
        refs_after = self.get_references_after_region(region, scope)
        used_after = {name for name, _ in refs_after}

        # Return variables that are both defined in region and used after
        return_vars = sorted(defined_in_region & used_after)
        return return_vars

    def get_function_body_region(self, func_node: Node) -> Region | None:
        """Get the region covering the function body."""
        if func_node.type != "function_definition":
            return None

        body = func_node.child_by_field_name("body")
        if body is None:
            return None

        return Region.from_node(body)

    def get_indentation(self, node: Node) -> str:
        """Get the indentation string for a node."""
        # Get the line containing the node
        lines = self.parser.source.split(b"\n")
        line = lines[node.start_point[0]]

        # Count leading whitespace
        indent = []
        for ch in line:
            if ch in (ord(" "), ord("\t")):
                indent.append(chr(ch))
            else:
                break

        return "".join(indent)

    def get_scope_indentation(self, scope: Node) -> str:
        """Get the base indentation for inserting at scope level."""
        return self.get_indentation(scope)

    def get_body_indentation(self, scope: Node) -> str:
        """Get the indentation for code inside a scope body."""
        base = self.get_indentation(scope)
        # Add one level of indentation (detect from source or default to 4 spaces)
        # Try to detect from the body if possible
        if scope.type == "function_definition":
            body = scope.child_by_field_name("body")
            if body and body.child_count > 0:
                first_stmt = body.children[0]
                return self.get_indentation(first_stmt)

        # Default: add 4 spaces
        return base + "    "
