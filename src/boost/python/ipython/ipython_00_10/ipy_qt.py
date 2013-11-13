# -----------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

import logging
import PyTango
import PyQt4.Qt as Qt

BRUSH = Qt.QBrush
COLOR = Qt.QColor
_WhiteBrush = Qt.QBrush(Qt.Qt.white)
_BlackBrush = Qt.QBrush(Qt.Qt.black)
_RedBrush = Qt.QBrush(Qt.Qt.red)
_GreenBrush = Qt.QBrush(Qt.Qt.green)
_DarkGreenBrush = Qt.QBrush(Qt.Qt.darkGreen)
_BlueBrush = Qt.QBrush(Qt.Qt.blue)
_YellowBrush = Qt.QBrush(Qt.Qt.yellow)
_MagentaBrush = Qt.QBrush(Qt.Qt.magenta)
_GrayBrush = Qt.QBrush(Qt.Qt.gray)
_DarkGrayBrush = Qt.QBrush(Qt.Qt.darkGray)
_LightGrayBrush = Qt.QBrush(Qt.Qt.lightGray)

ATTR_QUALITY_DATA = {
    PyTango.AttrQuality.ATTR_INVALID  : (BRUSH(COLOR(128, 128,  128)), _WhiteBrush),
    PyTango.AttrQuality.ATTR_VALID    : (_GreenBrush, _BlackBrush),
    PyTango.AttrQuality.ATTR_ALARM    : (BRUSH(COLOR(255, 140,   0)), _WhiteBrush),
    PyTango.AttrQuality.ATTR_WARNING  : (BRUSH(COLOR(255, 140,   0)), _WhiteBrush),
    PyTango.AttrQuality.ATTR_CHANGING : (BRUSH(COLOR(128, 160, 255)), _BlackBrush),
    None                              : (BRUSH(Qt.Qt.FDiagPattern), _BlackBrush)
}

DEVICE_STATE_DATA = {
    PyTango.DevState.ON      : (_GreenBrush, _BlackBrush),
    PyTango.DevState.OFF     : (_WhiteBrush, _BlackBrush),
    PyTango.DevState.CLOSE   : (_WhiteBrush, _DarkGreenBrush),
    PyTango.DevState.OPEN    : (_GreenBrush, _BlackBrush),
    PyTango.DevState.INSERT  : (_WhiteBrush, _BlackBrush),
    PyTango.DevState.EXTRACT : (_GreenBrush, _BlackBrush),
    PyTango.DevState.MOVING  : (BRUSH(COLOR(128, 160, 255)), _BlackBrush),
    PyTango.DevState.STANDBY : (_YellowBrush, _BlackBrush),
    PyTango.DevState.FAULT   : (_RedBrush, _BlackBrush),
    PyTango.DevState.INIT    : (BRUSH(COLOR(204, 204, 122)), _BlackBrush),
    PyTango.DevState.RUNNING : (BRUSH(COLOR(128, 160, 255)), _BlackBrush),
    PyTango.DevState.ALARM   : (BRUSH(COLOR(255, 140,   0)), _WhiteBrush),
    PyTango.DevState.DISABLE : (_MagentaBrush, _BlackBrush),
    PyTango.DevState.UNKNOWN : (_GrayBrush, _BlackBrush),
    None                     : (_GrayBrush, _BlackBrush),
}

def getBrushForQuality(q):
    return ATTR_QUALITY_DATA[q]

def getBrushForState(s):
    return DEVICE_STATE_DATA[s]

def deviceAttributeValueStr(da):
    return str(da.value)

ID, HOST, DEVICE, ATTRIBUTE, VALUE, TIME = range(6)
HORIZ_HEADER = 'ID', 'Host','Device','Attribute', 'Value', 'Time'

class EventLoggerTableModel(Qt.QAbstractTableModel, logging.Handler):
    
    DftOddRowBrush = Qt.QBrush(Qt.QColor(220,220,220)), Qt.QBrush(Qt.Qt.black)
    DftEvenRowBrush = Qt.QBrush(Qt.QColor(255,255,255)), Qt.QBrush(Qt.Qt.black)

    DftColHeight = 20

    DftColSize = Qt.QSize(50, DftColHeight), Qt.QSize(120, DftColHeight), \
                 Qt.QSize(160, DftColHeight), Qt.QSize(100, DftColHeight), \
                 Qt.QSize(120, DftColHeight), Qt.QSize(120, DftColHeight)
    
    def __init__(self, capacity=20, freq=0.1):
        super(Qt.QAbstractTableModel, self).__init__()
        logging.Handler.__init__(self)
        self._capacity = capacity
        self._records = []
        self._accumulated_records = []
        self.startTimer(freq*1000)

    # ---------------------------------
    # Qt.QAbstractTableModel overwrite
    # ---------------------------------
    
#    def sort(self, column, order = Qt.Qt.AscendingOrder):
#        if column == LEVEL:
#            f = lambda a,b: cmp(a.levelno,b.levelno)
#        elif column == TYPE:
#            def f(a,b):
#                if not operator.isMappingType(a) or not operator.isMappingType(b):
#                    return 0
#                return cmp(a.args.get('type','tau'), b.args.get('type','tau'))
#        elif column == TIME:
#            f = lambda a,b: cmp(a.created,b.created)
#        elif column == MSG:
#            f = lambda a,b: cmp(a.msg,b.msg)
#        elif column == NAME:
#            f = lambda a,b: cmp(a.name,b.name)
#        elif column == THREAD:
#            f = lambda a,b: cmp(a.threadName,b.threadName)
#        elif column == LOCALT:
#            f = lambda a,b: cmp(a.relativeCreated,b.relativeCreated)
#        self._records = sorted(self._records, cmp=f,reverse= order == Qt.Qt.DescendingOrder)
#        #self.reset()
    
    def rowCount(self, index=Qt.QModelIndex()):
        return len(self._records)

    def columnCount(self, index=Qt.QModelIndex()):
        return len(HORIZ_HEADER)
    
    def data(self, index, role=Qt.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._records)):
            return Qt.QVariant()
        column, row = index.column(), index.row()
        record = self._records[row]
        if record.err:
            err = PyTango.DevFailed(*record.errors)
        else:
            err = None
        name = record.s_attr_name.lower()
        if role == Qt.Qt.DisplayRole:
            if column == ID:
                return Qt.QVariant(row)
            if column == HOST:
                return Qt.QVariant(record.host)
            elif column == DEVICE:
                return Qt.QVariant(record.dev_name)
            elif column == ATTRIBUTE:
                return Qt.QVariant(record.s_attr_name)
            elif column == VALUE:
                if err is None:
                    return Qt.QVariant(deviceAttributeValueStr(record.attr_value))
                else:
                    return Qt.QVariant(err[0].reason)
            elif column == TIME:
                if err is None:
                    return Qt.QVariant(record.attr_value.time.strftime("%H:%M:%S.%f"))
                else:
                    return Qt.QVariant(record.reception_date.strftime("%H:%M:%S.%f"))
        elif role == Qt.Qt.TextAlignmentRole:
            if column in (HOST, DEVICE, ATTRIBUTE):
                return Qt.QVariant(Qt.Qt.AlignLeft|Qt.Qt.AlignVCenter)
            return Qt.QVariant(Qt.Qt.AlignRight|Qt.Qt.AlignVCenter)
        elif role == Qt.Qt.BackgroundRole:
            if column == VALUE:
                if err is None:
                    if name == "state":
                        bg = getBrushForState(record.attr_value.value)[0]
                    else:
                        bg = getBrushForQuality(record.attr_value.quality)[0]
                else:
                    bg = Qt.QBrush(Qt.Qt.red)
            else:
                if index.row() % 2:
                    bg = self.DftOddRowBrush[0]
                else:
                    bg = self.DftEvenRowBrush[0]
            return Qt.QVariant(bg)
        elif role == Qt.Qt.ForegroundRole:
            if column == VALUE:
                if err is None:
                    if name == "state":
                        fg = getBrushForState(record.attr_value.value)[1]
                    else:
                        fg = getBrushForQuality(record.attr_value.quality)[1]
                else:
                    fg = Qt.QBrush(Qt.Qt.white)
            else:
                if index.row() % 2:
                    fg = self.DftOddRowBrush[1]
                else:
                    fg = self.DftEvenRowBrush[1]
            return Qt.QVariant(fg)
        elif role == Qt.Qt.ToolTipRole:
            if err is None:
                return Qt.QVariant(str(record.attr_value))
            else:
                return Qt.QVariant(str(record))
        elif role == Qt.Qt.SizeHintRole:
            return self._getSizeHint(column)
        #elif role == Qt.Qt.StatusTipRole:
        #elif role == Qt.Qt.CheckStateRole:
        elif role == Qt.Qt.FontRole:
            return Qt.QVariant(Qt.QFont("Mono", 8))
        return Qt.QVariant()

    def _getSizeHint(self, column):
        return Qt.QVariant(EventLoggerTableModel.DftColSize[column])

    def headerData(self, section, orientation, role=Qt.Qt.DisplayRole):
        if role == Qt.Qt.TextAlignmentRole:
            if orientation == Qt.Qt.Horizontal:
                return Qt.QVariant(int(Qt.Qt.AlignLeft | Qt.Qt.AlignVCenter))
            return Qt.QVariant(int(Qt.Qt.AlignRight | Qt.Qt.AlignVCenter))
        elif role == Qt.Qt.SizeHintRole:
            if orientation == Qt.Qt.Vertical:
                return Qt.QVariant(Qt.QSize(50, 20))
            else:
                return self._getSizeHint(section)
        elif role == Qt.Qt.FontRole:
            return Qt.QVariant(Qt.QFont("Mono", 8))
        elif role == Qt.Qt.ToolTipRole:
            if section == HOST:
                return Qt.QVariant("tango host")
            elif section == DEVICE:
                return Qt.QVariant("tango device")
            elif section == ATTRIBUTE:
                return Qt.QVariant("tango attribute")
            elif section == VALUE:
                return Qt.QVariant("attribute value")
            elif section == TIME:
                return Qt.QVariant("time stamp for the event")
        if role != Qt.Qt.DisplayRole:
            return Qt.QVariant()
        if orientation == Qt.Qt.Horizontal:
            return Qt.QVariant(HORIZ_HEADER[section])
        return Qt.QVariant(int(section+1))
    
    def insertRows(self, position, rows=1, index=Qt.QModelIndex()):
        self.beginInsertRows(Qt.QModelIndex(), position, position+rows-1)
        self.endInsertRows()
    
    def removeRows(self, position, rows=1, index=Qt.QModelIndex()):
        self.beginRemoveRows(Qt.QModelIndex(), position, position+rows-1)
        self.endRemoveRows()

    #def setData(self, index, value, role=Qt.Qt.DisplayRole):
    #    pass
    
    #def flags(self, index)
    #    pass
        
    #def insertColumns(self):
    #    pass
    
    #def removeColumns(self):
    #    pass
    
    # --------------------------
    # logging.Handler overwrite
    # --------------------------

    def timerEvent(self, evt):
        self.updatePendingRecords()

    def updatePendingRecords(self):
        if not self._accumulated_records:
            return
        row_nb = self.rowCount()
        records = self._accumulated_records
        self._accumulated_records = []
        self._records.extend(records)
        self.insertRows(row_nb, len(records))
        if len(self._records) > self._capacity:
            start = len(self._records) - self._capacity
            self._records = self._records[start:]
            self.removeRows(0, start)
    
    def push_event(self, evt):
        attr_name = evt.attr_name
        dev, sep, attr = attr_name.rpartition('/')
        if dev.startswith("tango://"):
            dev = dev[8:]
        if dev.count(":"):
            # if it has tango host
            host, sep, dev = dev.partition('/')
        else:
            host = "-----"
        evt.host = host
        evt.dev_name = dev
        evt.s_attr_name = attr
        self._accumulated_records.append(evt)

    def clearContents(self):
        self.removeRows(0, self.rowCount())
        self._records = []
        self._accumulated_records = []

    def getEvents(self):
        return self._records
    
class EventLoggerTable(Qt.QTableView):
    
    DftScrollLock = False
    
    """A Qt table that displays the event logging messages"""
    def __init__(self, parent=None, model=None, designMode=False):
        super(EventLoggerTable, self).__init__(parent)
        self.setShowGrid(False)
        self.resetScrollLock()
        model = model or EventLoggerTableModel()
        self.setModel(model)
        hh = self.horizontalHeader()
        hh.setResizeMode(HOST, Qt.QHeaderView.Stretch)
        self.setSortingEnabled(False)
        #self.sortByColumn(TIME, Qt.Qt.AscendingOrder)

    def rowsInserted(self, index, start, end):
        """Overwrite of slot rows inserted to do proper resize and scroll to 
        bottom if desired
        """
        for i in range(start,end+1):
            self.resizeRowToContents(i)
        if start == 0:
            self.resizeColumnsToContents()
        if not self._scrollLock:
            self.scrollToBottom()

    def setScrollLock(self, scrollLock):
        """Sets the state for scrollLock"""
        self._scrollLock = scrollLock
    
    def getScrollLock(self):
        """Returns wheater or not the scrollLock is active"""
        return self._scrollLock

    def resetScrollLock(self):
        self.setScrollLock(EventLoggerTable.DftScrollLock)

    def clearContents(self):
        self.model().clearContents()
    
    def getEvents(self):
        return self.model().getEvents()

    def sizeHint(self):
        return Qt.QSize(700, 400)
    
    #: Tells wheater the table should scroll automatically to the end each
    #: time a record is added or not
    autoScroll = Qt.pyqtProperty("bool", getScrollLock, setScrollLock, resetScrollLock)


class EventLoggerWidget(Qt.QWidget):
    
    def __init__(self, parent=None, model=None, designMode=False):
        super(EventLoggerWidget, self).__init__(parent)
        self._model = model or EventLoggerTableModel()
        self.init(designMode)
        
    def init(self, designMode):
        l = Qt.QGridLayout()
        l.setContentsMargins(0,0,0,0)
        l.setVerticalSpacing(2)
        self.setLayout(l)
        
        table = self._logtable = EventLoggerTable(model = self._model, designMode=designMode)
        tb = self._toolbar = Qt.QToolBar("Event logger toolbar")
        tb.setFloatable(False)
        
        self._clearButton = Qt.QPushButton("Clear")
        Qt.QObject.connect(self._clearButton, Qt.SIGNAL("clicked()"), table.clearContents)
        tb.addWidget(self._clearButton)
        l.addWidget(tb, 0, 0)
        l.addWidget(table, 1, 0)
        l.setColumnStretch(0,1)
        l.setRowStretch(1,1)
    
    def model(self):
        return self._model

    def getEvents(self):
        return self.model().getEvents()

EventLogger = EventLoggerWidget