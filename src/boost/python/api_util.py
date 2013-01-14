################################################################################
##
## This file is part of PyTango, a python binding for Tango
## 
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
## 
## PyTango is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## PyTango is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################

"""
This is an internal PyTango module.
"""

__all__ = ["api_util_init"]

__docformat__ = "restructuredtext"

from ._PyTango import ApiUtil

from .utils import document_method, document_static_method, _get_env_var

def __init_api_util():
    if not hasattr(ApiUtil, "get_env_var"):
        ApiUtil.get_env_var = staticmethod(_get_env_var)

def __doc_api_util():
    
    ApiUtil.__doc__ = """
    This class allows you to access the tango syncronization model API.
    It is designed as a singleton. To get a reference to the singleton object
    you must do::
    
        import PyTango
        apiutil = PyTango.ApiUtil.instance()
        
    New in PyTango 7.1.3
    """
    
    document_static_method(ApiUtil, "instance", """
    instance() -> ApiUtil

            Returns the ApiUtil singleton instance.

        Parameters : None
        Return     : (ApiUtil) a reference to the ApiUtil singleton object.

        New in PyTango 7.1.3
    """ )

    document_method(ApiUtil, "pending_asynch_call", """
    pending_asynch_call(self, req) -> int

            Return number of asynchronous pending requests (any device). 
            The input parameter is an enumeration with three values which are:
            
                - POLLING: Return only polling model asynchronous request number
                - CALL_BACK: Return only callback model asynchronous request number
                - ALL_ASYNCH: Return all asynchronous request number

        Parameters :
            - req : (asyn_req_type) asynchronous request type
            
        Return     : (int) the number of pending requests for the given type

        New in PyTango 7.1.3
    """ )
    
    document_method(ApiUtil, "get_asynch_replies", """
    get_asynch_replies(self) -> None

            Fire callback methods for all (any device) asynchronous requests 
            (command and attribute) with already arrived replied. Returns 
            immediately if there is no replies already arrived or if there is 
            no asynchronous requests. 

        Parameters : None
        Return     : None

        Throws     : None, all errors are reported using the err and errors fields
                     of the parameter passed to the callback method.
        
        New in PyTango 7.1.3
        
    get_asynch_replies(self) -> None

            Fire callback methods for all (any device) asynchronous requests
            (command and attributes) with already arrived replied. Wait and
            block the caller for timeout milliseconds if they are some
            device asynchronous requests which are not yet arrived.
            Returns immediately if there is no asynchronous request.
            If timeout is set to 0, the call waits until all the asynchronous
            requests sent has received a reply.

        Parameters :
            - timeout : (int) timeout (milliseconds)
        Return     : None
        
        Throws     : AsynReplyNotArrived. All other errors are reported using 
                     the err and errors fields of the object passed to the 
                     callback methods.
        
        New in PyTango 7.1.3
    """ )

    document_method(ApiUtil, "set_asynch_cb_sub_model", """
    set_asynch_cb_sub_model(self, model) -> None

            Set the asynchronous callback sub-model between the pull and push sub-model.
            The cb_sub_model data type is an enumeration with two values which are:
            
                - PUSH_CALLBACK: The push sub-model
                - PULL_CALLBACK: The pull sub-model

        Parameters :
            - model : (cb_sub_model) the callback sub-model
        Return     : None

        New in PyTango 7.1.3
    """ )

    document_method(ApiUtil, "get_asynch_cb_sub_model", """
    get_asynch_cb_sub_model(self) -> cb_sub_model

            Get the asynchronous callback sub-model.

        Parameters : None
        Return     : (cb_sub_model) the active asynchronous callback sub-model.

        New in PyTango 7.1.3
    """ )

def api_util_init(doc=True):
    __init_api_util()
    if doc:
        __doc_api_util()