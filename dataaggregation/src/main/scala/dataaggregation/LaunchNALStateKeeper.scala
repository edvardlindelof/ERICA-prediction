package dataaggregation

import akka.actor._

import scala.collection.mutable

import sp.EricaEventLogger._
import sp.gPubSub.API_Data.EricaEvent

object LaunchNALStateKeeper extends App {
  val system = ActorSystem("DataAggregation")

  val evHandler = new NALStateKeeper
  system.actorOf(Props(new Logger(evHandler)), "EricaEventLogger")
}

class NALStateKeeper extends RecoveredEventHandler {

  val priorities = List("PRIO1", "PRIO2", "PRIO3", "PRIO4", "PRIO5")
  val teams = List("NAKME", "NAKM", "NAKKI", "NAKOR", "NAKBA", "NAKÖN") // "valid clinics" according to ericaBackend code
  val invalidTeams = List("NAKIN", "NAK23T", "NAK29", "NAKKK") // TODO should I include these??
  val patientKinds = List("all", "MEP", "triaged") ::: priorities ::: teams ::: invalidTeams
  // sets are of CareContactIds
  val patientSets: Map[String, mutable.Set[Int]] = patientKinds.map(str => str -> mutable.Set[Int]()).toMap

  override def handleEvent(ev: EricaEvent) = {
    println(ev.Start + "  " + patientKinds.map(kind => kind + ": " + patientSets(kind).size).mkString(", "))
    ev.Category match {
      case "RemovedPatient" => patientSets.foreach(t => t._2.remove(ev.CareContactId))
      case "Q" => patientSets("all").add(ev.CareContactId)
      case "T" => patientSets("all").add(ev.CareContactId)
      case "P" => patientSets(ev.Type).add(ev.CareContactId)
      case "ReasonForVisitUpdate" if ev.Value == "MEP" => patientSets("MEP").add(ev.CareContactId)
      case "TeamUpdate" => patientSets(ev.Value).add(ev.CareContactId)
      //case s: String if s.contains("removed") => patientsAtNAL.remove(ev.CareContactId)
      case _ => ()
    }
    ev.Type match {
      //case "KLAR" => patientsAtNAL.remove(ev.CareContactId)
      case "TRIAGE" => patientSets("triaged").add(ev.CareContactId)
      case _ => ()
    }
  }
}