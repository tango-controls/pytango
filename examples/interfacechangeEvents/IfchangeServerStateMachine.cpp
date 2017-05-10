/*----- PROTECTED REGION ID(IfchangeServerStateMachine.cpp) ENABLED START -----*/
static const char *RcsId = "$Id:  $";
//=============================================================================
//
// file :        IfchangeServerStateMachine.cpp
//
// description : State machine file for the IfchangeServer class
//
// project :     
//
// This file is part of Tango device class.
// 
// Tango is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// Tango is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with Tango.  If not, see <http://www.gnu.org/licenses/>.
// 
// $Author:  $
//
// $Revision:  $
// $Date:  $
//
// $HeadURL:  $
//
//=============================================================================
//                This file is generated by POGO
//        (Program Obviously used to Generate tango Object)
//=============================================================================

#include <IfchangeServer.h>

/*----- PROTECTED REGION END -----*/	//	IfchangeServer::IfchangeServerStateMachine.cpp

//================================================================
//  States  |  Description
//================================================================


namespace IfchangeServer_ns
{
//=================================================
//		Attributes Allowed Methods
//=================================================

//--------------------------------------------------------
/**
 *	Method      : IfchangeServer::is_busy_allowed()
 *	Description : Execution allowed for busy attribute
 */
//--------------------------------------------------------
bool IfchangeServer::is_busy_allowed(TANGO_UNUSED(Tango::AttReqType type))
{

	//	Not any excluded states for busy attribute in read access.
	/*----- PROTECTED REGION ID(IfchangeServer::busyStateAllowed_READ) ENABLED START -----*/
	
	/*----- PROTECTED REGION END -----*/	//	IfchangeServer::busyStateAllowed_READ
	return true;
}

//--------------------------------------------------------
/**
 *	Method      : IfchangeServer::is_ioattr_allowed()
 *	Description : Execution allowed for ioattr attribute
 */
//--------------------------------------------------------
bool IfchangeServer::is_ioattr_allowed(TANGO_UNUSED(Tango::AttReqType type))
{

	//	Not any excluded states for ioattr attribute in read access.
	/*----- PROTECTED REGION ID(IfchangeServer::ioattrStateAllowed_READ) ENABLED START -----*/
	
	/*----- PROTECTED REGION END -----*/	//	IfchangeServer::ioattrStateAllowed_READ
	return true;
}


//=================================================
//		Commands Allowed Methods
//=================================================

//--------------------------------------------------------
/**
 *	Method      : IfchangeServer::is_Add_dynamic_allowed()
 *	Description : Execution allowed for Add_dynamic attribute
 */
//--------------------------------------------------------
bool IfchangeServer::is_Add_dynamic_allowed(TANGO_UNUSED(const CORBA::Any &any))
{
	//	Not any excluded states for Add_dynamic command.
	/*----- PROTECTED REGION ID(IfchangeServer::Add_dynamicStateAllowed) ENABLED START -----*/
	
	/*----- PROTECTED REGION END -----*/	//	IfchangeServer::Add_dynamicStateAllowed
	return true;
}

//--------------------------------------------------------
/**
 *	Method      : IfchangeServer::is_Delete_Dynamic_allowed()
 *	Description : Execution allowed for Delete_Dynamic attribute
 */
//--------------------------------------------------------
bool IfchangeServer::is_Delete_Dynamic_allowed(TANGO_UNUSED(const CORBA::Any &any))
{
	//	Not any excluded states for Delete_Dynamic command.
	/*----- PROTECTED REGION ID(IfchangeServer::Delete_DynamicStateAllowed) ENABLED START -----*/
	
	/*----- PROTECTED REGION END -----*/	//	IfchangeServer::Delete_DynamicStateAllowed
	return true;
}

//--------------------------------------------------------
/**
 *	Method      : IfchangeServer::is_iocmd_allowed()
 *	Description : Execution allowed for iocmd attribute
 */
//--------------------------------------------------------
bool IfchangeServer::is_iocmd_allowed(TANGO_UNUSED(const CORBA::Any &any))
{
	//	Not any excluded states for iocmd command.
	/*----- PROTECTED REGION ID(IfchangeServer::iocmdStateAllowed) ENABLED START -----*/
	
	/*----- PROTECTED REGION END -----*/	//	IfchangeServer::iocmdStateAllowed
	return true;
}


/*----- PROTECTED REGION ID(IfchangeServer::IfchangeServerStateAllowed.AdditionalMethods) ENABLED START -----*/

//	Additional Methods

/*----- PROTECTED REGION END -----*/	//	IfchangeServer::IfchangeServerStateAllowed.AdditionalMethods

}	//	End of namespace
