package sp

import akka.actor._

import sp.EricaEventLogger._
import sp.gPubSub.API_Data.EricaEvent

object Launch extends App {
  val system = ActorSystem("DataAggregation")

  //system.actorOf(Props(new Logger()), "EricaEventLogger")

  val printer = new EricaEventPrinterCustommm
  system.actorOf(Props(new Logger(printer)), "EricaEventLogger")
}

class EricaEventPrinterCustommm extends RecoveredEventHandler  {
  override def handleEvent(ev: EricaEvent) = println("EricaEventPrinterCustommm received " + ev)
}
