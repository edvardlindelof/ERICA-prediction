package sp

import akka.actor._
import sp.EricaEventLogger.Logger

object Launch extends App {
  val system = ActorSystem("DataAggregation")

  system.actorOf(Props[Logger], "EricaEventLogger")
}
