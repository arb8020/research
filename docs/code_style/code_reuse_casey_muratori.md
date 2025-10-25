https://www.youtube.com/watch?v=ZQ5_u8Lgvyk&t=1s
code reuse is something that in general everyone wants to be able to do um either in-house or externally you
typically want to be able to solve a problem once or license you know a solution from somebody and just
integrate into your game and have it work um it's kind of a goal that's common across many Industries not just
the game industry um and so there it's no surprise that there's lots of books on it there's lots of people who speak
on the topic this sort of thing um but in general uh what seems to typically be the case is that a lot of
the things that you hear about this or think you know about this when you actually try to put them into practice
it turns out that it doesn't really work that well and you kind of sit there scratching your head going like you know
we thought that we were doing s of this code reuse stuff well or we we thought this component that we looked at to license had a good API but when when we
actually try to use it in our game we kind of like start to feel like getting a little more NIH about the problem
because it just didn't work out that well CA caused all sorts of problems and so on um so where this lecture kind of
comes from is uh I worked at rad game tools and I was responsible for doing granny which is the character animation
package um and at the outset of that I kind of made the first version of it according to what I thought were
standard you know reasonable code PR principles uh and it turned out to just
not really work very well you know a lot of people had a lot of trouble integrating it um it was just it you know things just didn't go the way that
I thought they should so so we spent a lot of time uh when we went from 1.0 to 2.0 kind of looking at that and going
you know what what happened there and we learned a lot of things and the second incarnation of the product actually is
used in I think 400 different SKS um which is a vast vast increase from the original one so with this suer I kind of
tried to go back and go okay what were all the things we learned about that um and how did you know how did we make
that jump are they generally applicable and are there are there like things that we can kind of generalize from this that
are real rules we can use to get reuse in the game industry um that are from practice and not just some kind of
theoretical thing um that doesn't seem to really play out in practice so before I I get into it I

## Three types of reuse

just wanted to quickly go over the types of reuse that we typically see um the first kind here on the left is uh what I
call layer which is sort of like an open Gil or direct 3D kind of a thing where you've got new code that you're writing
on top which is your game or something um you've got reused code uh which is the layer which is the the you know
direct 3D part um or whatever the API is that's provided to abstract some service that's down at the bottom which is like
the 3D Hardware or something like this and this kind of reuse is is pretty common now and it seems to be working
relatively well um there's still problems with that sort of thing the second kind is engine code which is kind
of inverted it's like the reused code is up at the top uh making all the decision just the majority of the system and
you're just writing kind of this smaller subset of that um conforming to to its rules giving it information back um to
Define your game and it's pretty much responsible for for ending up dealing with all the output and that sort of thing and then finally we have the Third
Kind which is kind of the new thing um which hadn't been around really in any successful way up until very recently
which is components and those are things where it's still kind of like the layer situation where it's the new code is up
at the top you know the game code is is all in control um but there's kind of these little pieces that you can license
which not only in input from you but actually give uh stuff back to you that that actually determines how the game
runs they're tightly integrated and these are things like character animation physics AI that sort of thing
um and they don't really talk to Services directly there's no well-defined output layer for them they
just give you stuff back and then you're responsible for kind of doing it now uh the reason those are important and the
reason that layers aren't the only thing that we need I mean engines I'm not really going to cover because that's just you know a totally separate topic
um the reason that that components are important is because if you look at layers they're very limited um they

### Layers alone are insufficient

require there to be some well-defined kind of service at the bottom that they're abstracting like 3D Hardware um
and if they don't have something like that so if you want to do something like AI where there's no agreed upon standard for what the output is um then they just
you know they can't really exist and similarly if you have uh you know two layers that you're trying to use in the
game simultaneously well that just isn't going to work at all you know if I have two things that expect to be talking to the 3D Hardware that can be a huge
disaster cuz each one wants control over it and how do you broker between them so it's no longer this clean like I give
something to it it puts something out now components solve both of these problems they don't need a service at

### Components provide solution

the bottom they can just be there they get you know you give them data they process it they give you data back it's all good um and similarly if there's one
service you know one output that you're that you're dealing with that both services are cons that both components are concerned with that's fine because
the input is coming back into the game where you can broker between it resolve conflicts do anything you need um so
components are really like the most powerful form of subsystem reuse that there is um and it they're the ones that
we happen to need to solve a lot of the problems of reuse that we you know for components we don't have currently um the problem with them is
they're much harder to design and the reason for that uh is this little back channel here that I drew I mean it's
like a little simple arrow but in reality like the fact that it is coupled back to the game and you have to rely on
what it's giving you to make gameplay decisions to do all these things uh really does mean that it's a vastly
harder design problem than just output only um and not to say that you know direct three and open jail weren't hard
to design it's just these are even harder and that's unfortunate so uh what I'm going to do
here is for a couple slides I'm going to introduce kind of like a model for how to look at the way games integrate
things which may or may not be unique to how other Industries integrate things I don't have much experience with that but
for games there's a very specific way that integration typically happens with components so I'm going to go through a
little bit of that and uh then I'm going to go into some really uh very in-depth
code snippet kind of stuff where I take you through all of the different ways that kind of you can use this model to
look at what's happening in making decisions about either is some licensable API that I'm going to use uh
going to work well or I'm trying to design an API that I can reuse in house or for other developers you know what
can I do to make those apis better so here's my little uh sort of pseudo graph um the vertical

## Integration grows in spurts

axis here is how much work was done to integrate the particular uh API into our
game and down here is time and what some people like tend to gloss over if you
read books about this sort of thing is that integration isn't really something that you just sit down and do we don't
like one day decide to integrate physics into our game we spend a week on it and then we're done and that's the the last we hear of it that's not what happens
what really happens is yes that that part of the process is over here somewhere we do the initial integration
that's in the game but then in reality we get usually as we go we use a little bit more of it and then we get to some
point where we have some major push that we're trying to do and at that point we
typically have to start getting more involved with that API because there's things we need to do like for example we
have memory budgets we need to hit uh we need to add some features that we didn't think we were going to have to need or that we put off uh integrating and now
we need them for this demo we need them for whatever we're trying to ship and we get kind of this little Spike there and
similarly that typically happens again at least once at the end of the project uh when we're trying to ship there's all
of these hard constraints that we have to meet so it really is an evolving usage it goes from from initial where we
might even just be prototyping a game so we're not even doing the official integration all the way till like we're
trying to ship and nail down every little last piece okay so here's the kind of like

### Integration options

abstract part so try to put on your abstract cap what I want to do is get you in the in
the mindset of thinking about the options that you or other developers have for integrating a particular thing
into their game a particular component now I took time out of this
now so we're just looking at it a single point in time and the amount of work to integrate something is here and the
amount of benefit to the game is here and these little big blue circles are just representing options that the
developer has for integration so you know when I go to integrate something like character animation into my game
there's a number of ways I could do it I look at the game engine I look at the component and I go well we could do something like this we could wrap it
like this we could put it in over here we could have we could Fork it and put it part of it over here and over here so you've got a number of ways in which you
can integrate it and typically as you go you know the more integration work it is for you typically the more benefit you
could get because the tigh the more tighter coupled you become to this component the more things you can do
with it you know if you start managing its memory for it then you're going to get performance benefits and there's all these kind of things that as you go up
so what happens is when you initially integrated in you've got some minimum bar that you're trying to meet in terms

### Initial integration

of benefit you've got like we need these features for the game right now so we're just going to try and meet that and you
know hey you have particularly smart programmers so they're going to kind of go well this is kind of the thing that's
the least amount of work that gives me the most amount of benefit that meets the bar right but you know I could do a
little more work and maybe get something that I thought was a little more beneficial so maybe I will you know so there's kind of you're going to pick
something kind of down here to get it in get it working get the game running and get on with things then as you go at some point
there's going to be some new requirements you're going to need more benefits from this API and when I say benefits I don't necessarily mean

### Integration progression

features I just mean there are things that you need from it the ability to like get its memory footprint down the
ability to like reclaim some processor time those things are all considered part of this because you know again they're integral to shipping the game so
this bar moves across well you're kind of going to start going okay well we need to kind of integrate a little more we need to integrate a little more I
need to change what we're doing call some different apis do that sort of thing and you move you move through this
kind of space of possibilities that you had and then finally what happens if if you you know you get into trouble is you
get to some part where the requirements change and there is no easy jump to get you there right you have this thing that
you have to do and maybe it's like um some kind of hard hard budget that you have to meet and you realize that the
way that you were using the API there just is no simple fix there's no way that you can get say streaming which you
want to implement in your game into this component in any reasonable amount of work it's going to be this massive thing
where you have to do all sorts of stuff behind its back and do all these kind of things so you end up looking at a situation like this where you're like
wow to get this extra thing that we needed to do to ship this game we have to do a massive amount of work to deal
with this component so to kind of summarize that it's like hey if the API has these
little steps and I want to meet this minimum bar of features that I've got to get for this next drop I want to go from
unsolved to solve I just want to take a little step I want to do some work to get it whatever I feel is the minimum amount of work that I should generally
have to do to get this thing in there but if there is no options for that then they typically will have to jump all the
way to something which while you had to do a ton of work to get there it really you know solved a lot more problems than
you were looking to solve you ended up doing a lot more management or a lot more work with this component than you
would have liked maybe there are fringe benefits to that but a lot of times those benefits aren't really realized to
you you just wanted to manage one little piece of it for for the component for example but now you ended up having to
manage all of its memory let's say so I call that an integration

### Integration discontinuity

which is like I'm kind of going along using this component and all of a sudden I hit this wall and I'm like man this is a disaster and unfortunately
they typically happen around ship time just when it's like most unfortunate to have such a thing happen so
I guess uh this is this is kind of a little like uh beating a dead horse but just to go through it if you think about

### Discontinuities waste work

this yellow line as how much work you actually did to integrate the product meaning I'm actually doing this work and
I keep doing work and I do more work and I do more work when you have these discontinuities in there what happens is
you're really doing more work than the benefit that you were getting right so when I have to do more work than the
minimum that I wanted to do to get this feature in then the actual benefit to your app isn't going up commensurate
with the work you're like spending a lot of extra work to work around this API that you're forced to deal with and this
doesn't even have to be increasing because what could happen is if you're spending time integrating the component you may find out that the way that you
are integrating it isn't actually working and you have to like rip some of it out redo the way you were doing it
throw out code you wrote to reintegrate it a different way so you can actually end up going like okay we got to tear some out we tried this other method and
that didn't work so then we tried going around at this other way and we finally found an end run that worked and now we keep going and so on so I guess all I'm
trying to say with this graph is like it's it's non-trivial when when you have these like API uh problems in there it
really can force you to blow out a lot of work and um that's why I pause it
that's like the real problem the goal that we're trying to solve when we dissolve reusable apis is to think about
this and to try to make it so that at all times as people integrate the product they are always able to do only
what they think think they should have to do to get the next thing that they need out of the API and not waste all of
this time because really when they use code I mean from a licens perspective that's what the customer ends up remembering right they don't remember if
you save them some work at some point they remember when you cause them this huge disaster right around ship time
right and similarly if you're the person doing that you don't want that disaster so this is really the thing that I think
is most important now unfortunately uh the current trends in API design a lot you

## Current API design trends

know with things where you kind of wrapping stuff up doing a lot of encapsulation doing a lot of insulation
what they actually do is they take the number of options that are available to the developer which are these you know blue things like I said before and they
start to reduce them they go you know what you can't really have access to these things you can't call these things without calling these other things and
at some point sometimes you even get to the point where you look at an API and you're like you know what there's only one thing I can do with this like they've totally walled me off I have no
options anymore this is a disaster and these kind of components typically fail um but a lot of times we're stuck with
just this kind of thing where kind of there's a low level or high level way to use and so on um and I argue that really
like you know we want things to look like this and ideally fill in this Gap that I was talking about
before okay so uh that's kind of the end of the abstract part so now you get now
you're going to have to uh read some code snippets on here so hopefully can everyone like read this well enough to
see say that second me okay good so these are the characteristics

## Five Characteristics

that I have identified that I think are are uh sort of indicative of how that
those those blue points because that was very abstract you know I drew a graph had some blue points on it you know well that's all great but how do I actually
know when I'm looking at an API you know is it going to have these problems are there discontinuities in there um you
know how much options does it really give me and uh originally I only had four in fact if you look in the uh in
the in my lecture notes there were only four but uh I was showing some of the slides to Chris Hecker and he was like
you don't talk about FL control enough you got to talk about flow control it's it's implicit in all the things you're talking about you're not talking about
so I looked at it and he was totally right so I added a fifth one which kind of like does really uh it it does really
matter so Props to Chris for that the five things are in order uh granularity
which is to say that you know I have some API and I can split it into smaller apis so if I have an API a I can replace
it with two apis that do the things that it did but give me a little more control um there's redundancy which is to say
that there's an API that does something which is a but I could alternatively call some other API which does the same
thing B only maybe it takes slightly different parameters or it does it in a slightly different way so I have you
know some options there um there's coupling which is when you have one thing and if you do that thing in the
API you're required to do some other thing so there's like this hidden kind of Link in between them that you are not
really able to overcome um there's retention which is when the API kind of has two things and like one's on your
side and then the other's on their side you have to kind kind of mirror them like making a scene graph or something like this and finally there's flow
control which is like who's calling who am I calling them is the component calling me am I calling it and it's calling me back you know what's going on
here now the important thing to remember is these are just characteristics and each one of them has trade-offs which
I'll talk about later so it's not some of them are always bad but the majority of them have like sometimes it's good to
have have like less granularity it's good to have more so you know don't think of them as like hard-coded we want
all these five things it's like no there each one of them is is just a characteristic and we'll talk about how interpret them in a second so uh to go through the kinds of
granularity some of them are non obvious the most obvious thing is hey if I have
some like update orientation and what this function is supposed to do is the API has some measure of my orientation
that it's keeping and it's got some measure of the change in orientation and I want to go ahead and apply that change
so now when I use the orientation it's the new orientation well you know simple granularity change is okay I want to
break that down into steps I want to get the orientation myself I want to get the change in orientation myself from the
API and then I want to you know set the thing with that change and these are angles or something so that's not like
any kind of overloaded plus Operator just like you know hey it's like you know just a a regular 2D angle um that's
the most simplest kind of granularity the reason that I want that is because hey I may want to modify whatever is
going to happen in there right I may not want it to just use it directly um the change that it has I may want to play
with it right and similarly the kind of less obvious version of that is I may not ever want to change it I may want
this to happen exactly the same way that it would have if I just called it but I have this other thing that I want to
have happen so instead of like modifying things by inserting myself in the middle really all I'm doing is I'm separating
when the API is going to do those two things and that may not be that important in other Industries but in the game industry that's like crucial
because sometimes you thread things sometimes you have like kind of things that you need to hold over till the end of the frame so you really don't want to
be in positions where where you don't have that kind of control so that kind of granularity is also
important so okay uh so now let's talk about redundancy um because granularity
hopefully that's pretty clear so redundancy is just you know in its most basic form is something like this I
wanted to pass a 3X3 Matrix before and now I want to pass a quan so the API gives me two calls and I can enter in
either way and it just accepts the you know the type of perimeter that I was looking for doesn't do anything
different C and D are a different uh way of looking that which is that hey sometimes I figure that there should be
like these basic things that it can just do for me I don't want to have to make my own identity Matrix and pass or that sort of stuff there's just constants
that the API builds in that's easy for me to use and that way you know I just know it's taken care of and similarly
often times there are things that I would do with the orientation and I just want it to do those basic operations for
me to set the operation I mean to set the orientation rather and that's D so those are just some different redundant
ways of doing that now the sort of non the the sort of subtle way of of uh of
having redundancy is this kind here where if you remember in the previous slide I couldn't fit on the slide we had
this kind of operation where we're getting the orientation changing the Orient getting the change in the orientation and then setting it well if
I was to go up a level of granularity from that I could have the option of bundling those three calls in two
different ways I could bundle the first two calls and leave the third one at the low level I mean the the the um final
level of granularity or I could bundle the second two calls and leave this one at the finer level so they're both like
kind of equivalent they're at the same level of granularity these two Snippets but they have different choices in
redundancy in terms of they have different choices in what to bundle which makes kind of a redundant API there so that can typically be pretty
useful as you make coarser grained versions of an API to have the user have the ability to choose which ones they're
going to bundle which ones they're not so now we get to coupling which is not a trade-off thing coupling is pretty much

### Coupling - A implies B

always bad um but you know it's usually also unavoidable in a lot of places so
the simplest kind of coupling in an API is when you have something that does like a bunch things to lots of objects
and you have no control over that so you know a tip very typical thing is like simulate and a physics simulator where
it's just like hey you know I maybe I wanted some control over what was getting simulated because I have some special things that I want here but like
maybe this API doesn't let me do that so I have to just like have everything happen at once um okay so obviously that
that's bad coupling that's you know inter object coupling the other kind of coupling is hey I've got some apis which
kind of depend on this one state that I set so maybe I call set time and it you know retains this time information but
then like lots of different apis use that so I'm current creating a hidden sort of coupling between those apis in
the sense that like you know now they all have to kind of have this right ordering of if I set the time and then
call this I can't then call the other thing which counted on the time being thing from the previous room so I'm like kind of having this kind of like a
hidden coupling that I have to think about out in my head um the snippet C is the kind of coupling where you have in
like a GL begin GL end for example which is that there isn't any identification for a particular lock that I'm doing so
I can only have one of them at any given time you know I don't spe I don't really get anything back from this and I don't
pass anything into this so there's really just this one implicit lock that either I am or I'm not using um and that
kind of couples it in the sense that two pieces of code can't do that so you have to make sure that all of the code always just is serially doing something like
this is very simple which is hey if there are internal buffers are those internal
buffers going to be like things that we have to pay attention to so in this case if I was returning a carar for example
this is probably returning the same buffer as this so I've got this kind of hidden coupling of like string one actually becomes the same thing as
string two here because you know whatever hope hopefully that doesn't happen too much in modern apis but you know figured I'd mention it snippet e is
kind of a more Insidious form of coupling it's when the allocation of something is coupled to its initialization and a lot of API guys
have this problem unfortunately um a lot of times developers don't really want to have to be able to say oh you know what
get me the memory for this thing and initialize it they might want to go like I'm going to provide the memory and then could you just initialize it in place or
you know what you're managing the memory but I need to initialize this guy because I'm not reading it from the special stream that I have packed or whatever so that's a kind of coupling
where like I bundle two things together that cannot ever be separated um f is where I have coupling
between some special type in the system like if I only only accept a matrix for my orientation this is kind of gets back
to the rcy thing if that's the only type that I accept here then even if I have some like representation of the Matrix
myself I have to constantly like make it with the other you know with the API I have to say like hey make one of your
special fancy Matrix objects so that I can actually call your functions and the final kind of coupling is when you
depend on uh when when the API doesn't let you get away from depending on their file format so in this case it's like if
the only way to get an object is to read that object with their file reading routine right I can't construct it
myself with my own reading it in then I'm kind of dependent on their file IO routines and their data format and
there's nothing I can do about that really okay retention is pretty simple

### Retention - A mirrors B

so not many code slipit here uh the idea there is just hey if I have stuff that is you know data that I kind of own or
that I am the one who's like in charge of but the API forces me to announce that data to it and it keeps a copy
that's reten so in a it's the simplest kind I have you know I'm going to set the time or
I'm going to set what I think the value of pi should be for the application um then it just retains that information
going to use it everywhere um in B we have kind of like yeah well you know what I'm going to tell you that this
object is parented to this other object so every time you do something like up toate the orientation you're going to
take that into consideration and then finally in D um we have the kind where
you're retaining services from the application so the API is kind of going you know what when I open a file I could
call you back with some of these things so it's going to retain the services that you provided and use them whenever
it would have used them in the middle of processing some of its various function
calls and finally we have flow control and flow control is uh pretty easy to imagine pretend these are just stack

### Flow Control - A invokes B

traces and Float control is just you know a measure of flow control anyways it's like who is calling who um is it
the case that I you know the game is on the bottom of the stack then it calls into the library and that's always the
way it looks that's all we get so library on top game on bottom or do we have a situation where hey the game
which you know was originally calling the library now gets called back and there is sort of library in between game
on either side of the stack then we could get you know totally crazy and say well a lot of times then
the game has to call the library something so we can keep add INF an item if we allow this kind of flow control stuff to happen we can get these
ridiculous Stacks where it's like I call the libr calls me back I call the library maybe it calls me back one more time now this is obviously you know kind
of a negative thing because the more this happens the more kind of complex it is to visualize in your head what's
going on your relationship to this library and furthermore uh it can be really kind of nasty where it has to
call back one of your classes or you have to have void Stars which tell it what the you know the data is that you're going to need inside there
because you no longer have your scope so you there's a lot of complexity when you start introducing complicated kinds of flow control that aren't just a um and
the and the Cod Snippets for that are this is the most basic kind like we said before I just call a function and it
returns something everyone's happy B is the slightly more complicated version where it's going to call me back so you
know I do open file and I get a call back for it C is just up here because it's the same as B if anyone thinks that
c is not the same as B um definitely rethink that because hey this is just a function pointer right a virtual
function is just there's a v table somewhere so if you're inheriting from one of the api's classes that's exactly
the same thing as setting some file call backs and then finally we have you know hey you could use exceptions or
something like this transfer float control but hopefully that's not like a big part of any licensable API so here's the

### Recap

recap we have granularity and essentially what we do with that is we trade off flexibility
for Simplicity the more coarser grain the granularity is the simpler the API is to use because there's less calls
less things for the user to get wrong etc etc but we give up flexibility as we go up there we're kind of at the point
where they're really kind of out of control you know they don't have the control they're kind of just telling you what to do and the API does it you know
entirely the way that they want with redundancy it's really a trade-off of convenience versus orthogonality um if
redundant if your thing is very redundant then you're going to be very convenient because there's tons of these apis to choose from for every different
thing to do uh but you're not very orthogonal right there it's kind of a little harder for the user to keep in their head because there isn't just one
way of doing the thing that they're going to do there's lots of different ways with coupling um there really isn't
much of a trade-off it's like the API we'd like it to be as decoupled as possible because all it's doing is putting restrictions on what the
programmer can and can't do so there's really no benefit to that one retention is kind of a synchronization versus automation thing where it's like if I
retain a lot of data then it's kind of hard because the uh the user has to synchronize all that stuff and that's
bad um so you know mediate mode things are better because there's no there isn't that synchronization involved where they have to constantly keep you
updated um but you know you may lose some automation there because if you have all this retained data then the app
the the API can maybe um automate things for you that it couldn't otherwise and finally flow control again not much of a
trade-off there uh if you can get away with just always having the game in control and it calls the app and it
returns to you well that's always simpler because you don't want to have to worry about these like kind of deep call back situations um or how you get
data down through the through the library to you on the other side and so on and uh the final thing that I want to
mention about that is just that looking at all of those trade-offs they're not necessarily constant throughout the

### Tradeoff decisions often vary

course of the integration when I first integrate a component into my game I'm probably looking for very low
granularity so I'm looking for a very coarse grained uh granularity approach and uh a lot of retention because I just
want to do something like say hey load some characters off dis and animate them walking around and like free them later
like that's kind of the level I'm looking for when I'm just like pre-production or you know doing my first integration into the game whatever
but as I get to the end of the project I typically need the reverse right I need a lot of control so I want fine grain
you know fine granularity in several places that I really need to manhandle and I don't want a lot of retention
because I've built all these data structures that say the way my game works and the less of that I have to Mirror On Your Side um on the API side
the better so uh is everyone clear on that is there any questions hopefully that
okay um so what I'm going to go through now is I'm going to go through some looking at actual code Snippets that uh
you know they're they're not exactly the same as as Coast Snippets in a game but they're very very similar to it so we
can look at what happens uh when some of these things are not at the proper level that they should be just to give you a feel for if you're designing API how you
should be looking at it or if you're evaluating an API looking at what the consequences of that API are so uh
obviously I've I've changed this this isn't none of these are are specifically somebody's API uh they're just like very
representative of the standard ones that are out there so first thing thing let's look at
game provided services so in this case um K kind of alluded to this in earlier slides here's a case where I want the um

## Game-provided services

the API uh I was just kind of calling this read file thing in the API and I was getting back whatever the the the
thing is that it gives me back some kind of object that I'm going to use um I want to stop it from from touching the
disc you know I'm I'm going to manage that because I'm reading from my own database format or something like that um so typically what most apis do I'm
assuming they provide this at all which hopefully they do um is we get something like snippet B which is that I set some
file callbacks which is basically just me saying I'm providing like fop F read and so on now um so just call me when
you would have called those and the problem with that not only is it a problem for flow control because hey now
I'm you know getting called back by the uh by the API but it's also a problem for coupling in kind of a non-obvious
way and the reason that there's coupling here is because you've bundled the concept of reading the file with
interpreting it into this object that you can use so I've actually kind of smacked two
separate things together one of which is loading an inter loading an actual like chunk of data off the disk and the other
is interpreting it so even if I allow the the user to do those operations I
haven't given them any control over when they happen because they're still going to happen right at this call and there's
no way to separate those two things out so um and you know if the user wanted to
actually separate them out what they have to do is make heinously complicated versions of these that do like cashing behind somebody's back or you know God
knows what's going to happen uh so the much more decoupled way and which most like very very few apis do but some do
is uh to kind of give you the ability to just pass in some file data it's like okay I already read it want you to go
ahead and interpret this into a thing for me right and that's kind of more what you want to see because if you want this kind of control typically you want
something that's a lot more like this where I'm just feeding it chunks and it's interpreting them into the type that I want now is that the most
decouple we can get you know it's not um you look at that and you go well this is
still something that's kind of owned by them right the thing that's coming back like came back from the API and I had no
control like that's got to have some memory somewhere like something's going on here um at the very least let's
pretend that that file data that it's interpreting is compressed in some way uh so at the very least it's got to
decompress it first before it can be used so what's happening inside this call is we you know the API is
allocating a buffer decompressing into it and then returning me a pointer to some part of that um well I could couple
it further right I can go like this which is to say that I want it to decompress this raw file data for me
into file data and then it can do it can you know make the thing for me I can use and then I can get rid of this file data
because I don't need it anymore um but the problem with that is hey you know what it's still allocating memory so
then I typically get right back to where I was before where I have to then give it more yet more callbacks um I you know
I was trying to eliminate those before but now I'm right back to it because I got to let it allocate the memory for me so finally we get to f which is more of
the like properly decoupled version which is to say that here now I've got you know a four-line version of what was
a oneline thing before but now I have complete control right I get the size that I will need for this thing I malok
that or you know new it or however you want to get it call your own special allocator do whatever you want um then I
decompress it and then I ask it to make the thing so now everything is entirely within my purview here and all it's
doing is translating uh you know a pointer to something else
now we can still get decoupled one further step this one is is not
necessarily necessarily all the time but for certain types of apis it's crucial and that is this is the snippet that we
had but what we really might want to do is something more like this where we're saying you know what I don't really even
want you to require me to call one of your functions before I can use some of this data I don't want your a you know
however your pack data format Works to influence that that um so what I what I want to do here is I want to just say
hey you know what give me you know make one of these things and I'm going to read it in however the hell I want to read it in you know um and that's kind
of important because maybe I then want to go ahead and control exactly where that thing is placed with my own allocators and read it directly in and
by the time we get to H we realize you know what we didn't need the API at all for this process right simple twool line
thing if the data is transparent to us we can just do this and have complete control just like it was something in
our game and we haven't really affected the way the API functions at all right all the rest of the API can still work exactly the same way but now we've kind
of removed it entirely from this process so again I'm not trying to suggest that the difference between a and H is like
always use H or anything like this I'm just saying like here is an example of the huge spread of ways in which we
could do this one thing and typically you do end up wanting to do this one quite a bit in an average game for you
know apis that are tightly integrated so a is good and a lot of times the user will call that but H is also good and
you don't want to be in a situation where the only thing you have is a so let's look at another common thing
which is parameter redundancy parameter redundancy like I said before is um the ability to kind of like call something

## Parameter redundancy

with two in two different flavors or multiple flavors and so on in a I have the uh the original version of the apis
function which inverts some transform and let's pretend that the Transformer passing and the inverse we get back are
both in their like that's their data type that's their object or whatever it is um well the problem is when we start
realizing that hey we have our own transform type and we got to kind of pass that in um we end up with with
something like B where like maybe I have like a floating Point uh Vector of position and a floating Point Vector that's the rotation and now I've got to
call the API to bundle it up into one of their things and then when I'm done with their function I got to call Copy
transform to kind of you know get it out again to into my format and that can get really heinous if my version the you
know the the uh the the the object type that I'm using for a Transformer also has this process that has to happen
then I've got to get it out of mine put it into theirs call their function get
it out of theirs then put it back into mine and like that seems kind of silly because like all we were doing is this
and now we have to do all this so like you know seems like kind of a contrived example but at the same time like I see
this all the time in actual code like this sort of thing happens all the time it's not like this kind of random things so it's like you're always currying back
back and forth between these types and it ends up just being this kind of mess so really what you want to do is get as
close as possible to D um which is where maybe the API has several versions of this function because it knows that
everyone has their own format for position and or orientation so let's try to provide as many of them as we can to
reduce the chances that they have to do this kind of ridiculous dance um so here we're kind of like saying well you know
if you've got a querian well you know pass that directly into us and we'll just kind of like take these two modify them and pass them back out to
you okay um so let's look at the transition between a coar grained uh a

## Granularity transitions

course grain operation and A fine grain one in code snippet a um I've got kind
of a retain mode thing going on here where the app has some some kind of object called a node which has some stuff in it and I'm asking it to update
it and then I'm asking it to render it two operations on the Node um I'm
highlighted this one because maybe all I really want to do is change the way that operation works the rendering is great
all the the rendering is happy but the updating is not doing it things the way I want so in code snippet B I've
expanded it out and since I'm not using the node structure anymore as a developer I now have to duplicate all of
the stuff that that thing did and I'm just kind of making up some stuff that it did here um the update part did those
three things and the render part did those five things as you can see as a developer I'm a little sad because I didn't really want to know all that
render stuff I didn't want to do any of that but I had to do it all because I wanted to stop using that node thing so
I could get control over the update part well that's not very preferable and if we look at that uh what we really wanted
to do is just kind of have the exact same call that was there before but now instead of using the bundled node type I
just pass the parts of that node type that I wanted um and this is a very easy thing for an API to do but unfortunately
almost none of them actually seem to do it a lot of times when they have these kind of bundled constructs they don't actually provide the exact same function
in an unbundled way which is really kind of unfortunate um and similarly I could
look at a further kind of granularity option which is if they did want a little bit of control over one of the
processes maybe I even offer another level of granularity in between those two where it's like not only do I allow
you to kind of go from node to non- node so I can break up these two things and not have to worry about this one but maybe I also allow you to do part of the
render in me now and pass in this additional part so I can sort of take a little bit of the process away from you
but you still do the rest of it and I don't have to worry about it okay and now here's the final example
we're going to look at um which is kind of the retain mode problem the typical retained mode API problem that we get

## Retention mismatch

into um when you don't provide some of the ability to do things immediately so I've used the physics engine here
because this is the they're kind of the thing that's most typically known for being very retained mode heavy uh snipp
it a is kind of what you would typically see in a physics API it's like I'm going to create this rocket and this pole
that's stuck in the ground and I'm going to create a joint between them anyone who's ever played rocket jockey knows where this comes from um and I'm going
to kind of like constrain the rocket to the pole so that as the rocket goes it kind of is circling around the pole and
then I call simulate and it kind of just all magically works and everyone's happy my Rocket's flying around is great but
what happens unfortunately in game development is you know that that's not really very representative of what happens what typically happens is I've
got something that's like some sampled thing that's very fine grain like is the user pressing the x button and what I have to do if this is
what my API looks like is I got to go oh well if they're pushing the x button but then if I'm not already having one of
these hook lines in there I've got to make one and if I had one and they weren't pushing the button then I've got
to delete it and remember that I deleted it and then I can simulate so what I'm essentially doing there is I'm writing
almost the like diff for every part of my game where I diff their retain mode
version uh with what I actually know to be going on in my game and hopefully I did the diff properly and made all the
right calls to like change it up right um that is really heinous when you end
up working in an app like this like I think probably most people in the audience have had to deal with this at some point if they've ever used a
heavily retained mode API um what I'd much rather do is turn this ugly snippet that I don't like into
this snippet I mean this is all I was trying to do all I'm trying to do is say hey you know what do a joint between the
rocket and the pole when the X button's down like that's what I was trying to do so instead of writing this I want to be
able to write this and that's the real difference between immediate mode and retain mode is like immediate mode allows you to proceed Al do something so
you've got code code is very powerful you're making decisions in that code and you want to be able to just dish the output of that code directly to the API
and have it use it you don't want to have to go through a data intermediary which is the retained part of it which
you constantly have to worry about how to differentially update and the thing that I'd like to remind everyone here
because hey you know that I can only show so much on a slide um is that having a do joint call like this
immediate mode is much more powerful than if I say had just a Boolean that I could turn on and off in a retained mode
joint and the reason for that is what if I have no idea what pole the guy is going to catch this time well like I
don't want to have to do stuff like create speculative joints between all the rockets and all the poles and then just turn the Boolean on and off so I
can have the nice immediate mode feel right I don't want to do any of that stuff so the idea behind immediate mode is I've gone through the code I've
figured out exactly what I want to do I just want to call the API with it right there and I don't want to have to worry about having set up a retained mode
structure previously which will allow me to have that that code look the way I want which is what booleans in a in a
retain mode thing would do for you so that's not a
substitute um so finally you know now I've hopefully given you idea of how those five characteristics kind of work
when you look at the code um what ends up happening I think with most apis not
that there's that many that have tried kind of to do all this stuff but when you when you look at them I've talked to some people who have actually done a
bunch of this stuff and and they kind of seem to concur with me so when you kind of follow these either by Intuition or because you've sat and actually looked
at what all they they all are um the apis that work the best for reuse across multiple projects that are different
very different from each other um you know wide reuse kinds of situations um that people have a pleasant time with is
that when you optimize these five characters you get an API which kind of has gradual tiers to it right there's like lots of different kind of ways that
I can access this API and I can gradually move between them as I need to
they're highly decoupled so I may even have multiple tiers of stuff that don't even have to like kind of couple to each
other they're very separated if I decide to start getting more you know uh finer grained in one area then I don't have to
drag down everything there they're all kind of like staying at the level that I roughly want um they have no retain mode
stuff at their bottom so in other words at the very very finest level that I can
call this API it is not retaining data at least not that I know about if it is it's behind the scenes it's caching it's
stuff they're doing for performance but I am not responsible for doing that cash in for them and as you kind of go up
through the ti maybe there is some retain mode you know sometimes I do want to just say create rid of body create R Body create joint there's nothing wrong
with that but by the time I get down to the part where I'm I'm asking the API I want control here I don't expect to have
that be happening um and then finally uh i' I'd say that you you pretty much never want to be in the situation of
having the API dictate the flow control to you because most of the time there's there's really just no no reason for that so that's kind of what I see is
like the way apis look when you've really kind of got this stuff dialed and granted like you know like I was saying
with Granny one and Granny 2 I think we got a lot of the stuff in granny 2 but you know I look back at it now obviously
and there's tons of stuff I would do differently so it's not like I'm claiming that you know I've ever made an API which does all these things
perfectly but it definitely as as you kind of feel like you're making apis better I feel like they go towards this
and I think over the next few years we'll start seeing apis that kind of if they can do this we'll kind of get more
towards that kind of perfect point where all these things are working together properly
so um you know some of you may may not have wanted to spend a full uh 50 minutes talking about this stuff so you
know right now I've summarized it all into just like a minute's worth so you can get it all right now if you are
designing an API or if you're about to evaluate an API these couple slides should give you everything that you need
to know to cheat on the exam um the first and second things that you need to
know and this is kind of obvious but I just wanted to State it because most of the this lecture is about the more detailed stuff is just that always write
the usage code first you know I mean when you sit down to design an API

## API evaluation checklist

always write all of the like examples of using it that you can think of write those first don't start by like opening
up an H file and start writing a class declaration or something like that start by writing use user code and furthermore
if you've got a game sitting around which a lot of you do like pretend you're integrating it like pretend you have the magical API that does
everything you want go into that game and integrate it through and look at what you came up with that's the first
past API that's the best first past API you're going to get cuz all of your intuition is enabled there naturally and
you've got all the constraints of that game already playing into it so you've got a really good first example there
and furthermore if you're evaluating one you may think that the best thing to do is go read the documentation of the of
the tools that you're you know potentially evaluating foring don't do that yet pretend you have the perfect
one you want pretend to integrate it into the game you've got for a day or two look at what you came up with and
now when you evaluate those components from the different vendors go how close does this match what I'm going to do
like don't think in their terms first think in your terms first and then as you evaluate your apis go like which one
of these things you know links up with me so anyway most of the lecture was not on that but I figured it was very
important to mention because a lot of people won't do that so now we get to the stuff that hopefully this lecture
has argued is is the right thing to do the first thing is that any kind of retain mode construct that this API puts
forth I should be able to do the exact same thing in immediate mode um by just calling a function with the things that
the retain mode structure had in it right and that was the thing about the update node render node thing I should
always be able to kind of have the non- node version of those so I can just call them immediately and what that does is
that allows you to transition from the retained mode to a more finer controlled immediate mode when you need to if you
don't have those then when you need to transition you're going to have to go oh my God like what was that node thing doing I don't even know I got to go
learn about that whatever second thing anytime there's call backs or inheritance there should
be an equivalent way to just make an API call that doesn't have to use either of those things if you ever see something
that requires inheritance or requires you to use callbacks then that is giving up both flow control and it is coupling
you so you should never see that if you see something like that that's a red flag um third thing no API should ever
require you to use a specific data type of its own if you know full well that
your game and probably every other game in the universe has their own version of that data type like the last thing you
need is for you to go license three components from three different vendors they all use different Vector types and
you've got to use all three different Vector types in various places of your app because that's what they expect like that's the last thing you want to have
happen uh number four any API function that you would not consider Atomic for
purposes of writing your game if you think there's any way that you might not consider that thing Atomic at any time during integration then it should be
able to be replaced by between like say two and four different apis that replace it that are more granular right and that
doesn't count maybe accessor functions but just you know the actual physical operational calls um because if you look
at something and you go you know what that thing looks kind of a little high level to me and I don't see any way to
break it down further then that's a really good example of where you know some point deep in the ship process
you're going to have to hope you can get one of their API developers on the phone to kind of put in some back door access for you or something like that and
that's really bad um and the final four things uh any
data which doesn't have a reason for being opaque should be transparent um if you just have data structs that the API
is using for things those things should not be opaque right you always have the choice as a developer of not touching
the data structures of the API so if you're concerned about making you know
you know touching us kind of internal structure or something like that and having that break when the API gets
revved you can always decide not to do it right there's nothing forcing you to go look at you know some of the
structures that the that the API defines but when it comes down to shift time you
may find that you want that access to do some things in certain places in your app and if they aren't exposed to you
then you've kind of got this problem where you'd better hope the API has actual calls that do the sorts of stuff
you need directly or you're going to be in big trouble so the idea idea is you should look through it and while you
shouldn't expect to use all of those things you should try to make sure that they're there for when you do need them you should never have to use their
Resource Management which means like memory management file management string management so
on excuse me um and you should uh never have to use their file format and you should always be able to get full source
code to the runtime so those are pretty simple thanks sty
[Music] [Applause]
