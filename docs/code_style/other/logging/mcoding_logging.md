Intro
0:00
Welcome back, aspiring professional  Python devs to mCoding, I'm James Murphy.
0:04
Let's learn how to set up logging in  your python application, the modern way.
0:09
That means we're going to learn to stop  doing this, and start doing this instead.
0:15
As always with so-called "best  practices" these are general guidelines
0:19
that work for most use cases.
0:21
However, you know what's best for  your code way better than I do,
0:24
so if you have a good reason,  then do what you know is right.
0:27
Don't blindly follow this advice  if it doesn't work for you.
0:31
With that said, let's get going  ... right after I sponsor myself.
Self-sponsorship
0:34
At mCoding, we do code reviews,  consulting, and professional training.
0:38
Do you have a problem with your code?
0:39
Worried about your architecture?
0:41
Or do you just need some extra dev hours?
0:43
Why not make me part of your team?
0:45
Anyway let's get back to logging.
0:47
The built-in logging package is very old  and full of stuff that you shouldn't use.
logging is the standard
0:53
It doesn't follow PEP8, the docs are a mess,
0:55
and many tutorials are outdated or quite frankly  written by people who obviously don't use it.
1:01
But for all its quirks, the  built-in logging package is  
1:03
the de-facto standard for logging in Python.
1:06
It doesn't matter if you're using a cloud  provider or just developing a local application,
1:10
the standard is the built-in logging package.
1:12
But do you really even need logging at all?
1:15
Hey man, you came to me.
1:17
If your app doesn't need logging, that's up to  you, I'm just here to show you what's possible.
basicConfig
1:21
All right, let's see how to get started.
1:24
You might think you just import logging,  make a logger, set it up with a basic config,
1:28
and then get to logging with your messages.
1:30
When you run the code, based off the log level  you set, you see those level messages and worse.
1:36
And I wouldn't blame you if  you've been doing it this way.
1:39
In the most basic cases this is totally fine,  but I don't really recommend doing it this way.
1:45
Why? Well, if you care to do logging  at all you probably want to log
1:48
to at least two different places,  like to stdout and also to a file.
1:53
Or you want to log higher priority things  like errors and exceptions differently,
1:56
maybe you want to send an email if  there's an exception thrown in prod?
2:01
You can manually make handlers and  filters and add them to your loggers,
2:05
but, trust me, just don't do it that way.
dictConfig
2:08
Instead, use dictConfig.
2:11
For some reason dictConfig is hidden  away in this logging.config submodule,
2:16
and not so much as mentioned  in the main logging docs.
2:20
As the name suggests, this lets you  configure logging via a dictionary
2:23
that explicitly lists all of the necessary  components of your logging setup,
2:27
namely the filters, formatters,  handlers, and loggers.
2:32
The basic config hides these objects  and their relationships from you,
2:35
which makes for a really slick  one-liner in the most basic case,
2:39
but for more useful logging setups I find  it's much clearer to list them explicitly.
the complete logging picture
2:44
It can get really confusing if you don't have a  
2:46
good mental model for what  these things actually do.
2:48
So here's the one picture to keep in  mind to see how it all fits together.
2:53
Loggers are the things that  you actually use in your code.
2:56
They have a ".info" and you can  call logger.info to log a message,
2:59
logger.debug to log a debug message,  
3:02
logger.exception to log the  in-flight exception, etc.
3:06
That creates a log record,  
3:08
which is an object that contains all  kinds of useful contextual information.
3:12
Things like the message and  severity, the current time,
3:14
the current thread or async task, the  location in the source code, and so on.
3:20
Every logger can set a level to drop  all messages below a certain severity,
3:24
and optionally some filters to drop  or alter messages along the way.
3:28
So you could do something like drop all  messages that begin with "some annoying string."
3:32
Or you could do something more sophisticated  
3:34
like sensor private user data that  isn't supposed to appear in logs.
3:39
The logger then passes these log records  off to each of their handlers one at a time.
3:45
Handlers tell you how and where  to log a record, like to stdout,  
3:48
to a file. over email. or  to a commercial log service.
3:53
Each handler receives the log record and,  just like for loggers, the handler has a level
3:58
and some filters that allow it to drop  or alter the message along the way.
4:02
if a record is dropped by a handler, it  still passes on to the rest of the handlers.
4:06
But if it was dropped by the logger  itself, then it's dropped for good.
4:12
Assuming the message passes  through the levels and filters,
4:14
when it comes time for the handler to actually  write the log message, it needs to write text.
4:19
But currently this is a Python object.  So each handler has a formatter
4:23
that it uses to convert the log record  object into a string to be sent out.
4:28
The formatter is what lets you customize  what an individual message looks like.
4:32
Like is it the log level and then the message?  Or level, then timestamp, then message?
4:37
Or maybe you write JSON, or maybe  you write some other format.
4:41
Formatters are typically where  you see the most customization
4:44
because it's the formatter  that selects which data from  
4:46
the log record to actually include in the message
4:49
and that depends a lot on your specific use  case and what you want to see in your logs.
4:53
And this is almost the complete picture.
4:56
Except this is the picture for the root logger,  the root as in the root of the tree of loggers.
5:06
Loggers are accessed and created by name,
5:08
and if you split the name by dots then  you end up with a tree of loggers.
5:13
So the A.X logger is a child of the A  logger, which is a child of the root.
5:18
By default, once a child's done handling a log  record, it passes that record up to its parent.
5:23
So if this log record was generated  down here in the A.X logger,
5:27
all of A.X's handlers would run,  then it would propagate up to A,  
5:32
and all of A's handlers would run,
5:34
then it would propagate up to the root  and all of the roots handlers would run.
5:38
This is done to make it easier for users  to disable messages from whole subsystems
5:42
just by disabling certain loggers.
5:45
Once again, if a record is dropped by  a handler it will continue moving on,
5:48
to include propagating up to the parent.
5:51
But if it's dropped by a logger,  then it stops and doesn't propagate.
5:55
But slow down there, this is way more  flexibility than you would usually need.
6:01
Having all these handlers and filters  and propagation at different levels
6:05
is unnecessarily complex for most use cases.
6:08
And it often leads to subtly broken configs.
6:11
So here's what I recommend.
6:13
See all these non-root handlers? Delete them.
6:17
Unless you've got a good reason,  put all handlers on the root logger.
6:21
This is simpler, but also having all  handlers on the root logger ensures
6:25
that any messages generated by third-party  libraries get logged and formatted
6:29
the same way as messages  generated by your own application.
6:33
Filters? Same deal.
6:35
There's a decent chance you  don't need any filters at all,  
6:38
but if you do put them all on the root logger
6:40
or their handlers.
6:43
Leave propagation turned on, which is the default,  
6:45
so that all messages propagate  up to the root logger.
6:49
However, don't use the root logger in your code.
6:55
If you use any of the top-level logging functions  like logging.info, this uses is the root logger.
7:00
So don't use any of those functions.
7:04
Make sure to use your own logger, which you get by  
7:06
using logging.getLogger and passing in  the name of the logger that you want.
7:10
This will create the logger first  if it doesn't already exist,
7:13
then you can use your logger.info  instead of logging.info.
7:18
And remember, your logger  doesn't have any handlers on it.
7:20
We're depending on propagation to  send all events up to the root logger
7:24
and have the root logger  actually handle the events.
7:27
If you have a small- to medium-sized application,  a single non-root logger is all you need.
7:31
If you have a very large application,  
7:33
then you should make one non-root logger for  each major subcomponent of your application.
7:39
You definitely don't need a logger for every file.
7:41
That'd be a waste because these are globals  that live for the entire life of the program.
7:46
With that all in mind, let's get back  
7:47
to configuring logging for a few  common setups using dictConfig.
Example: log to stdout
7:51
As a baseline let's just have a simple  config that logs everything to stdout.
7:55
If you're ever confused about the config,  
7:57
draw it out like this and then use  the picture to fill in the config.
8:02
This "version" is required  and the only valid value is 1.
8:05
This is so that they can change everything  in the future without breaking old code.
8:08
"disable_existing_loggers" does what it says,  
8:10
it disables anything that's not  explicitly listed in this config.
8:14
I'm going to go ahead and set this to false so  that I can get log messages from third-party code.
8:20
No filters in this configuration,  so let's just delete that.
8:25
Next define a formatter named "simple"  and give it a simple format string.
8:29
We didn't specify what class this formatter is,  
8:31
so by default it just uses the  built-in logging.Formatter.
8:35
It accepts the format string like this, using  this kind of weird printf-style format string.
8:40
Yes it's kind of weird but just deal with it or,  
8:44
foreshadowing, wait a minute  and we'll see a better way.
8:49
If you want to customize your own  format you can find a list of all  
8:51
the available variables in  the logging documentation.
8:55
Next we need to define the single stdout handler,  so we create this single handler named "stdout"
9:02
and set our "simple" formatter as  the formatter for this handler.
9:06
In order to get it to actually  log to stdout, we set its class to
9:10
the built-in logging stream handler  with a stream of sys.stdout.
9:14
The "ext://" slash here means "external",  
9:18
as in this is a variable that's  defined outside of this config.
9:22
And voila! In just 16 lines we've configured  what the basic config did in one line.
9:29
I know, I know, but keep in mind this more of  
9:31
verbose style is going to be a lot  clearer when we have more going on.
9:35
So stick with me.
logging JSON/YAML config
9:37
Despite the fact that we're using dictConfig,
9:39
that doesn't mean that we need to  keep the logging configuration as  
9:42
a literal dictionary in our Python source.
9:45
Nothing wrong with that,  but many find it convenient
9:47
to keep the logging config in a  separate file in JSON or YAML format.
9:52
Personally I prefer to keep my config in JSON, so  go ahead and create a JSON version of your config,
9:57
then load that JSON when  when your application starts.
10:01
If you wanted to use YAML instead,  
10:02
it would look basically the same except  of course you would have a YAML config
10:06
and you would "import yaml" and do  a YAML load instead of a JSON load.
10:10
Let's bring those side by side just  so you can see the two configs.
10:14
Obviously YAML is a lot more condensed, but I  find it to be a lot more error-prone as well.
10:19
And also there's no YAML parser built into Python,
10:22
whereas there is a built-in JSON parser.
10:25
So if you wanted to, you would have  to pull that in as a dependency.
10:28
"pyyaml" is a popular choice.
10:31
Keeping the log config in a separate  file also allows you to let your users  
10:34
adjust the logging config to their preference.
10:36
You know, if you trust your  users to do that kind of thing.
Example: errors to stdedd and all to file
10:44
Second setup. Let's modify the  config so that errors go to stderr,
10:48
and then all logs go to a file.
10:51
Change the "stdout" handler to "stderr"  and set its level to "WARNING".
10:57
Then, create a new handler and set  its class to a RotatingFileHandler.
11:01
A rotating file handler keeps appending logs  to a file until it reaches a certain size,
11:05
in this case 10 kilobytes.
11:08
After it reaches 10 kilobytes, it  creates a backup and starts a new file.
11:12
After three backups it starts  deleting the oldest one.
11:16
10 kilobytes is a pretty small limit, this is  just so that you can see the rollover happen.
11:20
You probably want to pick a few megabytes.
11:35
After running the script a bunch of  times, you can see it eventually created
11:38
this "my_app.log.1" and then  started using "my_app.log" again.
11:43
We're still using the "simple" formatter here.
11:45
But since we're saving to a log file,  why don't we include some extra details?
11:49
We accomplish that by adding this new "detailed"  
11:51
formatter and setting it as the  formatter for the "file" handler.
11:55
We include much more information in the format  
11:56
string, and we're also showing  off the "datefmt" format here,
11:59
which allows us to customize  how dates are printed.
12:02
Pro tip: use an ISO-8601-compliant format  and include the timezone. Trust me.
12:09
This way our log contains a lot  more useful contextual information.
12:13
For a lot of applications,  this is a great place to stop.
12:17
But, if you really care about  the quality of your log data,  
12:21
then I really suggest making one crucial change.
12:25
Take a look at this log file.
12:26
Glancing over it I can visually distinguish  different messages from each other,
12:31
but notice that we have tracebacks in here.
12:34
And what if log messages had newlines in them?
JSON logs
12:37
If I wanted to parse this programmatically  I'd need to be able to parse back out
12:40
all of the data that I put into it.
12:42
But it's kind of just in free-form text with  newlines that could be anywhere at this point.
12:47
That's kind of intractable.
12:50
The solution? Store persistent  logs in JSON format,  
12:54
so that they can be parsed easily later on.
12:56
This is a change in how to convert  a log record into a string,
13:00
so that's the job of the formatter,  we need a JSON formatter.
13:04
But oh wait, there's no  built-in logging JSON formatter.
13:10
There are a few you can pip install,  but let's just write our own.
13:13
Supposing we did, you'd think you'd  be able to just set the "class" key  
13:17
here and then pass in whatever arguments here,
13:20
and those will be keyword  arguments to the constructor.
13:23
That's what's happening here  with the handler, right?
13:26
Uhh, nope.
13:27
You can use your own class  here using the "class" keyword,
13:31
but if you do then all of the keys are  hardcoded to be the ones that the built-in uses/
13:38
So I could use "format" and "datefmt",  but I couldn't create my own "fmt_keys".
13:44
Why is it like this way?  Great question! Moving on.
13:48
Change "class" to "()" and then  it will do what you actually want:
13:53
call this and pass this as a keyword argument.
13:57
You'd have to do the same thing  anywhere else in the config,
13:59
like if I made my own custom handler that didn't  have the same interface as the built-in one.
14:05
Okay well with that weird road bump  out of the way, let's continue.
14:08
So we're going to pass in these format keys,
14:10
which is going to be a dictionary where the key is  the key that I want to appear in the log message,
14:17
and the value here, like "levelname",  
14:19
is the variable that we're going  to look up from the log record.
14:24
Okay so let's go ahead and write this class.
14:27
We're in a new file here and we just  inherit from the built-in loging Formatter.
14:31
Nothing special in the init, we just store  the format keys that we get from the config.
14:36
Then we need to define this "format" function.
14:38
This is a thing that takes the  record and goes to a string.
14:41
I'm using "@override" here to indicate that  we're overriding something from the parent class.
14:45
It's not strictly necessary, but it's a  good habit to get into marking these things.
14:49
All we do is extract out the record data into a  
14:51
dictionary and then use the "json"  module to dump that to a string.
14:56
As far as actually extracting those  fields goes, it's pretty simple.
14:59
Regardless of the config, I chose to include the  
15:01
message and a timestamp in  ISO format in UTC timezone.
15:07
We pull in any exception data if  it's present using some parent  
15:10
methods in order to extract things out nicely.
15:13
And for the rest of the keys we just grab  them from the attributes of the record.
15:17
It's pretty straightforward and you  could probably do whatever you want here.
15:21
And donzo! Update the config to use the  new JSON formatter and we're good to go.
15:28
Check our log file and we  see nicely formatted JSON.
15:32
A slight warning though,  this file is not valid JSON.
15:35
Each line is valid JSON.
15:37
This format is called JSON Lines and  the common file extension is ".jsonl".
15:42
So to parse it you just read the file  line by line and parse each line as JSON.
Extra context with the extra param
15:47
And double pro-tip: now  that we're outputting JSON,
15:50
it's actually really easy to add  lots of extra contextual information.
15:54
To do this we can just use the "extra"  argument in one of our log calls.
15:58
Give it a dictionary of extra information and  Python will stuff that onto the log record.
16:02
Then just update your formatter to pull in those  
16:04
extra attributes and now any  extras will appear in our JSON.
16:10
Here's the {"x": "hello"}.
Custom filter
16:13
If you're getting too many logs and you'd  like finer control over which ones to drop,
16:17
then you might need a filter.
16:20
The process of creating a custom filter is  very similar to creating a custom formatter.
16:24
Inherit from the built-in Filter, then define  your own override of the "filter" function.
16:29
Given a record, you return a bool to indicate  whether or not that record should be processed.
16:33
So this non-error filter does kind of the  opposite of setting the level to "INFO".
16:38
Setting the level to "INFO" would mean  that you would only keep messages that  
16:41
were "INFO", "WARNING", "ERROR", "CRITICAL",
16:45
but setting this non-error filter would  give you "DEBUG" and "INFO" instead.
16:49
You can also alter the record here,  
16:51
like if you wanted to censor private  data or return an altered copy.
16:55
I'm not going to deal with filters for the rest  of the video, but here's a homework exercise.
16:58
Using this non-error filter, create a logging  config that shows normal messages to stdout,
17:05
but error messages to stderr.
17:07
You're going to need the  filter to prevent duplicates.
A glaring flaw
17:11
Okay okay, surely there are no more glaring  flaws with this logging setup, right?
17:17
At the risk of using a forbidden word  in Python let's talk about performance.
17:22
By its very nature, calling a  log function results in I/O.
17:26
If a user makes a request to my web app  and that results in 10 log messages,
17:31
I don't want to add 10 round-trips worth of time  to my logging service before I respond to my user.
17:37
But currently that's what will happen because  all logging calls are synchronous and blocking.
17:42
The solution? Use a QueueHandler  to log off the main thread.
Log off the main thread with QueueHandler
17:47
Collecting log data isn't the slow part.
17:49
The slow part is sending  it wherever it needs to go.
17:53
A queue handler stores your log  records in a queue without blocking,
17:56
while an associated queue  listener accepts those messages
18:00
and passes them off to other  handlers in another thread.
18:04
In order to configure this, create  a new queue handler in your config.
18:08
The class is "logging.handlers.QueueHandler"  and then it accepts another list of handlers.
18:13
These are the handlers that it dispatches to,  
18:15
so basically take the handlers that  you had on the root handler before,
18:19
put them here and then change the queue  handler to be your only handler on the root.
18:26
This "respect_handler_level" for  for some reason is by default false,
18:30
which results in the behavior of sending every  message to every handler regardless of log level,
18:36
so yeah, that's probably not what we want.
18:38
I'm going to set this to true  so that it does what you expect.
18:42
There's one more thing we need to  handle over here in the main file,
18:45
which is that because the queue  handler is starting a thread,
18:48
that's not something that's  going to happen automatically.
18:50
When we set up our logging we need  to manually start that thread.
18:54
We accomplish this by getting the  queue handler by name and then
18:57
if it exists we start its listener's thread.
19:01
We also register an atexit callback in  order to call the listener's stop method
19:05
and shut it down gracefully when the program ends.
19:08
Alternatively, if you want to keep  all the work inside the config,
19:10
you could also subclass the queue handler  class and make it do this stuff in its init.
Success!
19:16
And success!
19:17
We finally have a high quality,  parsable, multi-destination,
19:20
non-blocking logging setup  for our Python application.
19:24
Feels good, doesn't it?
Library logging?
19:27
But notice I said logging for your  "application" not your "library".
19:32
Application authors know who their users are and  know what kind of logs that they want to see.
19:37
Whereas if you're writing library code,  
19:39
you don't know who your end user is and  you don't know what kind of logs they want.
19:44
Conclusion: for library code,  don't configure logging.
19:49
You can still use logging, create loggers,  log messages and other important events.
19:53
Just don't configure it with  dictConfig or any other config.
19:58
Let applications do the configuring.
20:00
If a user doesn't configure logging,  the default and expected behavior is
20:04
that warnings and above will be printed to stderr.
20:08
If a user of your library does configure logging,
20:10
then don't interfere with what  they want by adding handlers  
20:13
formatters or other things  that they're not aware of.
logging4p
20:16
Finally, do you remember log4j?
20:19
It's an extremely popular logging library for  Java that had a 0-day vulnerability wreak havoc
20:24
and cause absolute pandemonium in the business  world as thousands of large products and services
20:29
with millions or billions of users  instantly became vulnerable to an easy-to-do
20:34
remote arbitrary code execution vulnerability.
20:37
At the heart of the vulnerability was  a combination of logging user input,
20:41
combined with a plugin that allowed  loading remote data as Java object data.
20:46
Anyway here's Python's "makeLogRecord" function  that can be used to create log records manually.
20:51
For example "from a pickled  event received over the wire".
20:56
I'm not saying this is actively vulnerable.
20:59
but in-case "logging4p"  becomes a thing ... called it.
Thanks!
21:02
Thanks for watching, and  remember my company is mCoding.
21:05
So if you're still not satisfied with your  logging or other project setup, maybe we can help.
21:09
Don't be afraid to reach out.
21:12
Did you know that this entire channel  is funded completely by donations?
21:15
So huge thanks to everyone on patreon and  direct contributors for supporting the channel.
21:20
If you'd like to support the channel  go to patreon.com/mcoding and sign up.
21:28
Don't forget to subscribe and slap that  like button an odd number of times.
21:29
See you in the next one.
