subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d, workspace, workspace_size) bind(C, name='tsvc_2_s275_fp64')
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t, c_ptr
  use, intrinsic :: omp_lib, only: omp_get_max_threads, omp_get_num_procs
  implicit none
  integer(c_int64_t), value :: len_2d
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_size
  real(c_double), dimension(len_2d, len_2d), intent(inout) :: aa
  real(c_double), dimension(len_2d, len_2d), intent(in) :: bb
  real(c_double), dimension(len_2d, len_2d), intent(in) :: cc

  integer(c_int64_t) :: i, j
  real(c_double) :: prev
  logical, save :: diag_done = .false.
  integer :: maxth

  if (.not. diag_done) then
     diag_done = .true.
     maxth = omp_get_max_threads()
     write(*, '(A,I0,A,I0,A,I0)') 'DIAG nprocs=', omp_get_num_procs(), &
         ' maxthreads=', maxth, ' len2d=', len_2d
     flush(6)
  end if

  if (workspace_size < 0) then   ! unreachable: keeps the reserved pair referenced
     len_2d = int(transfer(workspace, 0), 8) + workspace_size
  end if

  !$omp parallel do default(none) shared(aa,bb,cc,len_2d) private(i,j,prev) schedule(static)
  do i = 1, len_2d
     if (aa(i, 1) > 0.0d0) then
        prev = aa(i, 1)
        do j = 2, len_2d
           prev = prev + bb(i, j) * cc(i, j)
           aa(i, j) = prev
        end do
     end if
  end do
  !$omp end parallel do
end subroutine tsvc_2_s275_fp64
