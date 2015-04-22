#define TAGTYPE std::string
#define CONTAINER std::set<std::string>
#include "profile.h"
extern "C" void Tau_metadata(const char * name, const char * value);
extern "C" void Tau_pure_start(const char* label);
extern "C" void Tau_pure_stop(const char* label);
extern "C" void Tau_set_node(int i);
extern "C" void Tau_get_context_userevent(void **handle, const char* label);
extern "C" void Tau_context_userevent(void * handle, double value);
using std::cout;
using std::strcpy;
////////////////////////////////////
//Functions used for profile test //
////////////////////////////////////
struct aThread{
std::map<const char *, void *> usereventMap;
std::set<std::string> trackSet;
std::ofstream fp;
int regionCounter;
int depthCounter;
int maxDepthCounter;
char stackTop[256];
};

struct aThread thisThread;
//[NTS]: For many of these calls we'll need some form of "context." By which I mean we'll need to be able to know what variables we're inducting on at a given point, as well as what we're tracking at a given moment. Perhaps "start" can take two additional parameters: a set of new induction variables and a set of events to track. In much the same way, "stop" might take a set of the same things to stop following (or inducting on). [Closed: we discussed this in your office, decided to add the induction variables we're interested in to the event requests. 

void HI_profile_init(const char* progName,const char* analysisFile) {
  thisThread.depthCounter=2; //little leeway, really this should be 1, but this gives us the ability to see a library call. Used by the analysis system to decide how big a call path to generate
  thisThread.maxDepthCounter=2;
  thisThread.fp.open(analysisFile);
  thisThread.regionCounter=0;
  strcpy(thisThread.stackTop,"OPENARCROOT");
	int i;
	//[NTD] What should we pass to the following function?
	Tau_set_node(0);
  Tau_pure_start(progName); //[NTS 7/26] If you want to make this fancy, make this label the entire cmd line used to launch the program (for example "Program: ./a.out myarg1="puppies are" myarg2="awesome"). The context often helps.
  Tau_metadata("OpenArc Profile Format","true");
}

void HI_profile_track(const char* label, const char* metricsOfInterest, const char* inductionVariables, bool isRegion)
{
    char tempstring[256];
    if(!isRegion)
      sprintf(tempstring,"%s::%s::%s::::%s\n",label,metricsOfInterest,inductionVariables,thisThread.stackTop);
    else
    {
      sprintf(tempstring,"%s::%s::%s::::%s\n",label,metricsOfInterest,inductionVariables,thisThread.stackTop);
      strcpy(thisThread.stackTop,label);
    }
    TAGTYPE tstring=tempstring;
    char metadataTag[256];
    //thisThread.fp<<(*start);//FILE
    if(!(thisThread.trackSet.count(tstring)))
    {
      sprintf(metadataTag,"%s%d","OpenArc Profile Region ",thisThread.regionCounter++);
      Tau_metadata(metadataTag,tempstring);
      thisThread.trackSet.insert(tstring);
    }
    
}
void HI_profile_shutdown(const char* progName)
{
  char depthAmount[10];
  sprintf(depthAmount,"%d",thisThread.maxDepthCounter);
  Tau_metadata("OpenArc Profile Depth",depthAmount); 
  Tau_pure_stop(progName);
  thisThread.fp.close();
}
// [NTD] what is the rule for naming label?
void HI_profile_start(const char* label) {
  if(++thisThread.depthCounter>thisThread.maxDepthCounter)
    thisThread.maxDepthCounter=thisThread.depthCounter;  
	Tau_pure_start(label);
}

void HI_profile_stop(const char* label) {
  thisThread.depthCounter--;
	Tau_pure_stop(label);
}
/**
double HI_profile_get_userevent(HI_userevent_t uevent);
	double value;
	//Convert the user event to corresponding double value.
	//[NTD] We have to come up with user events that user can choose.
  //[NTS] Don't know that this is actually necessary anymore. The value should be something numeric, think it should just work.
	switch(uevent) {
		case HI_uevent_cachemiss :
			break;
		default :
			break;
	}
	return value;
}
*/
// [NTD] what is the rule for naming label?
// We can suppose that each "profile measure"/"profile track" directive
// has label provided either by user or by the compiler.
void HI_profile_measure_userevent(const char* label, double value) {
	void * eventhandle=NULL;
	if( thisThread.usereventMap.count(label) > 0 ) {
		eventhandle = thisThread.usereventMap[label];
	} else {
		Tau_get_context_userevent(&eventhandle, label);
		thisThread.usereventMap[label] = eventhandle;
	}
	//Measure user event.
	Tau_context_userevent(eventhandle, value);
}
