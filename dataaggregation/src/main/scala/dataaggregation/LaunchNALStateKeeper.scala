package dataaggregation

import akka.actor._

import scala.collection.mutable

import sp.EricaEventLogger.Logger
import sp.gPubSub.API_Data.EricaEvent

object LaunchNALStateKeeper extends App {
  val system = ActorSystem("DataAggregation")

  val evHandler = system.actorOf(Props[NALStateKeeper], "NALStateKeeper")
  system.actorOf(Props(new Logger(evHandler)), "EricaEventLogger")

  scala.io.StdIn.readLine("Press ENTER to exit application.\n")
  system.terminate()
}

class NALStateKeeper extends Actor {

  // TODO this variable does not make sense yet
  val patientsAtNAL = mutable.Set[Int]() // set of VisitIds TODO maybe should be CareContactIds?

  override def receive = {
    case ev: EricaEvent => handleEricaEvent(ev)
  }

  def handleEricaEvent(ev: EricaEvent) = {
    println("patients at NAL: " + patientsAtNAL.size)
    ev.Category match {
      case "RemovedPatient" => patientsAtNAL.remove(ev.VisitId)
      case "Q" => patientsAtNAL.add(ev.VisitId)
      case s: String if s.contains("removed") => patientsAtNAL.remove(ev.VisitId)
      case _ => ()
    }
    ev.Type match {
      case "KLAR" => patientsAtNAL.remove(ev.VisitId)
      case _ => ()
    }
  }
}