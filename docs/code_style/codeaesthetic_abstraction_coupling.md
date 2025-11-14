0:00
You have a game where you have a main character, enemy characters and obstacles.
0:05
For each of these objects we need to have code that keeps track
0:09
of its position in the world and code for rendering the image for each object.
0:15
We could write the same code in all three classes,
0:18
but you recognize that this is all similar.
0:21
So instead you make a third class called Game Object that handles this for you.
0:26
You now have a general implementation that tracks and renders out the object.
0:31
The code is mostly identical, but you allow the subclass to specify
0:36
which image is shown.
0:39
You just created an abstraction.
0:46
Architects have gotten really good at this game
0:50
of identifying repetition and extracting it out.
0:53
We get into the mode of code repetition bad, more abstraction good.
0:58
But there's a hidden tradeoff that doesn't get considered: Coupling.
1:02
Most engineers conceptually understand what coupling is,
1:06
and they definitely feel its web when trying to modify an over coupled system.
1:10
But when designing software, you don't feel
1:13
the impacts of coupling.
1:17
I consider coupling to be an equal and opposite reaction of abstraction.
1:21
For every bit of abstraction you add, you've added more coupling.
1:26
Let's explore an example.
1:28
We have a program that saves out data to XML, but that's the old format.
1:33
So we want to move to JSON.
1:38
We could do this by adding a configuration to the save logic
1:41
to support both modes.
1:48
But this will make the removal
1:49
of XML complicated and dangerous
1:52
because they're all intermingled.
1:57
So instead we'll make the JSON writer a separate class.
2:00
Then we can just chop off the XML support by deleting the whole file,
2:04
not needing to unweave the program logic.
2:08
All right, so now we've written our fancy new JSON writing class.
2:12
We might notice that both take in a file name during construction.
2:16
Our little repetition detectors go off and realize that maybe.
2:21
Maybe we could extract this out.
2:24
So our instinct is to create a common class called file saver
2:28
that just takes the file name, and our subclasses can grab it
2:31
from this protected variable.
2:38
But this is a bad idea.
2:40
We've now coupled both of these classes to the same input.
2:44
They must take a file input.
2:46
So if there ever became a need to create something that didn't use a file
2:51
like a database or cloud connection, this would break the abstraction.
2:55
And on the flip side, this abstraction brings us no value.
2:59
What does this abstraction save us?
3:01
Well, I guess we don't need to assign the variable twice.
3:05
But this isn't any complicated logic.
3:07
It's simply assigning a variable.
3:09
So for me, this squarely fits into the not worth it camp.
3:14
Okay, now what about the save functionality?
3:18
We could consider creating an interface that represents the “save” method,
3:21
but we do know that this would increase coupling because now
3:24
both of these classes are constrained to the same save method.
3:29
So what benefits does this abstraction bring us?
3:32
Well, let's look at the usage.
3:35
We have an if statement that decides which class
3:37
to create and calls save on one of them.
3:40
So if we add a common interface,
3:42
we only get to remove this one duplicate save line.
3:46
That doesn't simplify the program in any meaningful way.
3:50
So I'd also put this into the it's not worth it camp.
3:53
At this point it's better to keep these as two distinct classes
3:57
with no connection at all.
4:01
There are two cases where it would make me decide it was worth it.
4:05
One would be if we added more save options.
4:08
If we had three or more we might want to extract the construction
4:11
of these save objects into a separate piece of code, especially
4:17
if the different savers had different parameters, like a database configuration.
4:23
The other case would be if we needed our program to defer
4:27
or repeat saving at a different point in the program.
4:30
For example, if we wanted to save every 5 minutes
4:33
automatically, we’d create a class called “IntervalSaver”,
4:37
and it would make sense for this Interval Saver class
4:40
to be unaware which saver it's calling.
4:43
In both cases, it becomes worth it when we want to separate the decision
4:46
of which saver we want from the time we actually want to save.
4:52
Overall, it's good to only apply abstraction
4:55
when the value it brings overweighs the coupling.
4:58
This does mean that you might have a little bit of code repetition.
5:01
But I think that a little code repetition
5:03
brings less pain when changing code than over coupling.
5:07
What do you think?
