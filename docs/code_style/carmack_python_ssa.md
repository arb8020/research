When I started working in python, I got lazy with “single assignment”, and I need to nudge myself about it.

You should strive to never reassign or update a variable outside of true iterative calculations in loops. Having all the intermediate calculations still available is helpful in the debugger, and it avoids problems where you move a block of code and it silently uses a version of the variable that wasn’t what it originally had.

In C/C++, making almost every variable const at initialization is good practice. I wish it was the default, and mutable was a keyword.
