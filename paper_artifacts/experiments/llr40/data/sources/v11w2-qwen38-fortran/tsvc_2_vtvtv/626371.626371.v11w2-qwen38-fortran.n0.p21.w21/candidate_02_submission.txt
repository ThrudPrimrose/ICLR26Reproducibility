module tsvc_2_vtvtv_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
contains
  subroutine do_chunk(pa, pb, pc, start, end_)
    implicit none
    real(c_double), dimension(:), intent(inout) :: pa
    real(c_double), dimension(:), intent(in)    :: pb, pc
    integer(c_int64_t), intent(in) :: start, end_
    integer(c_int64_t) :: i
    do i = start, end_
      pa(i) = pa(i) * pb(i) * pc(i)
    end do
  end subroutine do_chunk

  subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d) bind(C, name="tsvc_2_vtvtv_fp64")
    type(c_ptr), value, intent(in) :: a, b, c
    integer(c_int64_t), value, intent(in) :: len_1d
    real(c_double), dimension(:), pointer :: pa, pb, pc
    integer(c_int64_t) :: n, i, t, nt, start, end_, chunk
    n = len_1d
    call c_f_pointer(a, pa, [n])
    call c_f_pointer(b, pb, [n])
    call c_f_pointer(c, pc, [n])
    !$omp parallel default(none) shared(pa,pb,pc,n) private(i,t,nt,start,end_,chunk)
    t     = omp_get_thread_num()
    nt    = omp_get_num_threads()
    chunk = (n + nt - 1) / nt
    start = t * chunk + 1
    end_  = min(start + chunk - 1, n)
    call do_chunk(pa, pb, pc, start, end_)
    !$omp end parallel
  end subroutine tsvc_2_vtvtv_fp64
end module tsvc_2_vtvtv_mod
