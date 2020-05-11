import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk

class Handler:
	def onQuit(self, *args):
		Gtk.main_quit()

	def onSwitch(self,switch,data):
		if(data==True):
			print("on")
		else:
			print("off")

builder = Gtk.Builder()
builder.add_from_file("blinker.glade")
builder.connect_signals(Handler())

window = builder.get_object("windows")
window.show_all()

switch = builder.get_object("switch")


Gtk.main()