package sp

import akka.actor._

import sp.EricaEventLogger.Logger
import sp.gPubSub.API_Data.EricaEvent

object Launch extends App {
  val system = ActorSystem("DataAggregation")

  //system.actorOf(Props(new Logger()), "EricaEventLogger")
  val printer = system.actorOf(Props[EricaEventPrinterCustommm], "EricaEventPrinter")
  system.actorOf(Props(new Logger(printer)), "EricaEventLogger")

  scala.io.StdIn.readLine("Press ENTER to exit application.\n")
  system.terminate()
}

class EricaEventPrinterCustommm extends Actor {
  override def receive = {
    case ev: EricaEvent => println("EricaEventPrinterCustommm received " + ev)
  }
}
